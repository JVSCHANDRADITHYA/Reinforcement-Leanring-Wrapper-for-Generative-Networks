import os
import torch
import numpy as np
from options.test_options import TestOptions
from models import create_model
from util import util
import cv2
from datetime import datetime

# ---------------- RL -----------------------
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---------------- BM3D DENOISE -----------------------
from bm3d import bm3d


# ---------- NO-REFERENCE QUALITY SCORE ----------
def nr_quality_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = lap.var()
    noise = np.std(lap)

    brightness = np.mean(gray)
    brightness_penalty = abs(brightness - 128) / 128

    # strong noise penalty so RL chooses denoise
    score = (1.5 * sharp) - (3.0 * noise) - brightness_penalty
    return float(score)


# ---------- GAN INFERENCE (ONE-TIME ONLY) ----------
def gan_inference(model, image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    img = np.asarray([image_rgb]).transpose(0, 3, 1, 2)

    data = {"A": torch.FloatTensor(img), "A_paths": ["temp"]}
    model.set_input(data)
    model.test()

    out = model.get_current_visuals()["fake"]
    out = util.tensor2im(out)
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


# -------------------------------------------------------
# RL ENVIRONMENT (RL edits ONLY GAN output)
# -------------------------------------------------------
class EnhancementEnv(gym.Env):

    def __init__(self, gan_output_full, save_dir, obs_size=128):
        super().__init__()

        self.orig_full = gan_output_full.astype(np.float32) / 255.0
        self.current_full = self.orig_full.copy()

        self.save_dir = save_dir
        self.obs_size = obs_size

        # Actions: 0 bright+, 1 bright-, 2 contrast+, 3 contrast-, 4 denoise, 5 STOP
        self.action_space = spaces.Discrete(6)

        # Observation = flattened 128x128 image
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.obs_size * self.obs_size * 3,),
            dtype=np.float32
        )

        self.max_steps = 6
        self.steps = 0

    # ---------------- APPLY ACTION ----------------
    def apply_action(self, action):
        img = self.current_full.copy()

        if action == 0:
            img = np.clip(img + 0.07, 0, 1)

        elif action == 1:
            img = np.clip(img - 0.07, 0, 1)

        elif action == 2:
            img = np.clip(1.15 * (img - 0.5) + 0.5, 0, 1)

        elif action == 3:
            img = np.clip(0.85 * (img - 0.5) + 0.5, 0, 1)

        elif action == 4:
            # ---------------- BM3D (WORKS EVERYWHERE – NO PROFILE) ----------------
            img_uint8 = (img * 255).astype(np.uint8)
            img_f = img_uint8.astype(np.float32) / 255.0
            den = bm3d(img_f, sigma_psd=0.10)
            img = np.clip(den, 0, 1)

        elif action == 5:
            return img, True

        return img, False

    # ---------------- MAKE OBS ----------------
    def _make_obs(self, full_img):
        small = cv2.resize((full_img * 255).astype(np.uint8),
                           (self.obs_size, self.obs_size))
        return (small.astype(np.float32) / 255.0).reshape(-1)

    # ---------------- REWARD ----------------
    def compute_reward(self, prev_full, new_full):
        prev = (prev_full * 255).astype(np.uint8)
        new = (new_full * 255).astype(np.uint8)
        return nr_quality_score(new) - nr_quality_score(prev)

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        self.steps = 0
        self.current_full = self.orig_full.copy()
        return self._make_obs(self.current_full), {}

    # ---------------- STEP ----------------
    def step(self, action):
        prev = self.current_full.copy()
        self.current_full, stop = self.apply_action(action)
        reward = self.compute_reward(prev, self.current_full)

        # Save step image
        img_uint8 = (self.current_full * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.save_dir, f"rl_step_{self.steps}.png"), img_uint8)

        self.steps += 1
        terminated = stop or (self.steps >= self.max_steps)
        truncated = False

        return self._make_obs(self.current_full), reward, terminated, truncated, {}


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
if __name__ == "__main__":

    # ---- Create folder ----
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("results_RL", run_id)
    os.makedirs(save_dir, exist_ok=True)

    # ---- Load input image ----
    image_path = r"F:\RESIDE_data\train\hazy\2.jpg"
    original = cv2.imread(image_path)
    cv2.imwrite(os.path.join(save_dir, "original_input.png"), original)

    # ---- Load GAN ----
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.results_dir = save_dir
    opt.name = "desmoke"
    opt.model = "test"
    opt.no_dropout = True

    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # ---------------------- 1️⃣ RUN GAN ONCE ----------------------
    gan_output = gan_inference(model, original)
    cv2.imwrite(os.path.join(save_dir, "gan_output.png"), gan_output)

    # ---------------------- 2️⃣ RL ON GAN OUTPUT ----------------------
    env = EnhancementEnv(gan_output, save_dir, obs_size=128)

    agent = PPO("MlpPolicy", env, verbose=1, device="cpu")
    agent.learn(total_timesteps=2000)
    agent.save(os.path.join(save_dir, "rl_policy.zip"))

    # ---------------------- 3️⃣ RL INFERENCE ----------------------
    obs, _ = env.reset()
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    # ---------------------- 4️⃣ FINAL OUTPUT ----------------------
    final_img = (env.current_full * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, "rl_final_output.png"), final_img)

    print("\nALL RESULTS SAVED IN:", save_dir)
