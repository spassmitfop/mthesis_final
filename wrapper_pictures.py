import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ocatari_wrappers import MaskedBaseWrapperDict
from submodules.HackAtari.hackatari import HackAtari

folder = "imgs14"
l_wrappers = ["pixel_screen", "pixel_screen_color", "binary_mask", "binary_mask_kr", "binary_mask_1.2",
              "binary_mask_1.5", "pixel_screen_og", "planes", "planes_0.8", "planes_1.2", "planes_1.5", ]
game_names = ["Seaquest", "Boxing", "Freeway", "SpaceInvaders", "Amidar", "Pong"]
print_obj = True

for g in game_names:
    e = "ALE/" + g + "-v5"
    # env = OCAtari(e, frameskip=4, )
    env = HackAtari(e, frameskip=4, )  # modifs=["all_blue_cars"])
    env = MaskedBaseWrapperDict(env, buffer_window_size=4, extension=20)
    xd = [30, 50, 100, 150, 200]
    obs, info = env.reset()
    count = 0
    done = False
    while not done:
        count += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if count in xd:
            if print_obj:
                print(obs["obj"])
            for j in l_wrappers:
                obs_len = len(obs[j])
                if j == "pixel_screen_color":
                    for i in range(obs_len):
                        folder_path = f"{folder}\\{g.lower()}\\{j}"
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                            print(f"Folder '{folder_path}' created.")
                        img = obs[j][i]
                        image = Image.fromarray(img)
                        image.save(f'{folder}\\{g.lower()}\\{j}\\{j}_{count}_{i}.png')
                else:
                    for i in range(obs_len):
                        # plt.imshow(obs[j][i], cmap='gray')
                        # plt.title(f"{j}_{count}_{i}")
                        # plt.axis('off')
                        # plt.show()
                        folder_path = f"{folder}\\{g.lower()}\\{j}"
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                            print(f"Folder '{folder_path}' created.")
                        cv2.imwrite(f'{folder}\\{g.lower()}\\{j}\\{j}_{count}_{i}.png', obs[j][i])
        if count > max(xd):
            break
        done = terminated or truncated
