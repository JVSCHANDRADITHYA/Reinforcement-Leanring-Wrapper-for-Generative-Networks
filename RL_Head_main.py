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


# ---------- NO-REFERENCE QUALITY SCORE (PURE OPENCV) ----------
def nr_quality_score(img):
    """
    Higher = better image.
    Uses:
    - Sharpness via Laplacian variance
    - Noise estimate
    - Brightness penalty
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness via Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = lap.var()

    # Noise estimate
    noise = np.std(lap)

    # Brightness deviation penalty
    brightness = np.mean(gray)
    brightness_penalty = abs(brightness - 128) / 128  # 0 (good) to 1 (bad)

    # Combine heuristic score
    score = sharpness - noise - brightness_penalty
    return float(score)


# ---------------- GAN INFERENCE WRAPPER -----------------------
def gan_inference(model, image_np):
    """
    Runs your GAN model on uint8 BGR image and returns uint8 BGR output.
    """
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    img = np.asarray([image_rgb]).transpose(0, 3, 1, 2)

    data = {"A": torch.FloatTensor(img), "A_paths": ["temp"]}
    model.set_input(data)
    model.test()

    out = model.get_current_visuals()["fake"]
    out = util.tensor2im(out)
    out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    return out_bgr


# ---------------- RL ENVIRONMENT -----------------------
class EnhancementEnv(gym.Env):

    def __init__(self, gan_model, input_image):
        super().__init__()

        self.gan_model = gan_model
        self.orig_full = input_image.astype(np.float32) / 255.0
        self.current_full = self.orig_full.copy()

        self.obs_size = 128  # RL sees downscaled version

        self.action_space = spaces.Discrete(6)

        # RL OBS MUST BE UINT8 FOR CnnPolicy
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.obs_size, self.obs_size),
            dtype=np.uint8
        )

        self.max_steps = 6
        self.steps = 0

    # -------- actions --------
    def apply_action(self, action):
        img = self.current_full.copy()
        if action == 0: img = np.clip(img + 0.07, 0, 1)
        elif action == 1: img = np.clip(img - 0.07, 0, 1)
        elif action == 2: img = np.clip(1.15 * (img - .5) + .5, 0, 1)
        elif action == 3: img = np.clip(0.85 * (img - .5) + .5, 0, 1)
        elif action == 4:
            noise = cv2.fastNlMeansDenoisingColored((img * 255).astype(np.uint8),
                                                    None, 7, 7, 7, 21)
            img = noise.astype(np.float32) / 255.0
        elif action == 5:
            return img, True
        return img, False

    # -------- NR Reward --------
    def compute_reward(self, prev_img, new_img):
        prev = (prev_img * 255).astype(np.uint8)
        new = (new_img * 255).astype(np.uint8)
        return nr_quality_score(new) - nr_quality_score(prev)

    # -------- Build RL OBS (UINT8) --------
    def _make_obs(self, img):
        img_uint8 = (img * 255).astype(np.uint8)
        small = cv2.resize(img_uint8, (self.obs_size, self.obs_size))
        obs = np.transpose(small, (2, 0, 1))  # CHW
        return obs.astype(np.uint8)

    # -------- Reset --------
    def reset(self, seed=None, options=None):
        self.steps = 0
        self.current_full = self.orig_full.copy()
        return self._make_obs(self.current_full), {}

    # -------- Step --------
    def step(self, action):
        prev = self.current_full.copy()

        self.current_full, stop = self.apply_action(action)

        gan_out = gan_inference(
            self.gan_model,
            (self.current_full * 255).astype(np.uint8)
        )
        gan_out = gan_out.astype(np.float32) / 255.0

        reward = self.compute_reward(prev, gan_out)

        self.steps += 1
        done = stop or self.steps >= self.max_steps

        obs = self._make_obs(gan_out)

        return obs, reward, done, False, {}


# ---------------- MAIN PIPELINE -----------------------
if __name__ == "__main__":

    # ---- Paths ----
    image_path = r"F:\Derain_code_and_github\input_images\bench.png"
    results_dir = "./results/"
    model_name = "desmoke"
    model_type = "test"

    # ---- Load GAN model ----
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.results_dir = results_dir
    opt.name = model_name
    opt.model = model_type
    opt.no_dropout = True

    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # ---- Load Input Image ----
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("ERROR: Cannot load image.")
        exit()

    # ---- Create RL Env ----
    env = EnhancementEnv(model, input_image)

    # ---- Train RL Agent ----
    agent = PPO("CnnPolicy", env, verbose=1)
    agent.learn(total_timesteps=2500)

    # ---- Save RL Agent ----
    agent.save("rl_enhancer_knob")

    # ---- RL Inference ----
    obs, _ = env.reset()

    while True:
        action, _ = agent.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        if done:
            break

    final = (obs * 255).astype(np.uint8)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"rl_final_output_{timestamp}.png")
    cv2.imwrite(out_path, final)

    print(f"[RL] Final Enhanced Image Saved at: {out_path}")

    cv2.imshow("RL Final Output", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
