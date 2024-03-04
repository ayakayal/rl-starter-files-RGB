import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper
from PIL import Image


def make_env(env_key, seed=None, RGB='False', render_mode=None):
  
    env = gym.make(env_key, render_mode=render_mode)
    if RGB == 'True':
        env = RGBImgPartialObsWrapper(env)
       

    obs,_=env.reset(seed=seed)
    return env
def singleton_make_env(env_key, seed=10005, RGB='False', render_mode=None):
  
    env = gym.make(env_key, render_mode=render_mode)
    if RGB == 'True':
     
        env = RGBImgPartialObsWrapper(env)
    env.reset(seed=10005)
    #env = RGBImgObsWrapper(env)
    obs, _ = env.reset(seed = 10005)
    # # # Save the image using PIL
    image = Image.fromarray(obs['image'])
    image.save('image_env'+'.png')
    return env