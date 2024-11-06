import itertools


def combine_env_setting(env_names, settings):
    environments = []
    for env_name, setting in itertools.product(env_names, settings):
        environments.append({'env_name': env_name, 'setting': setting})
    return environments


g = -9.81
SCALED_GRAVITY = [
    {'gravity': 0.8 * g, 'name': '0.8g'},
    {'gravity': 0.9 * g, 'name': '0.9g'},
    {'gravity': 1. * g, 'name': '1g'},
    {'gravity': 1.1 * g, 'name': '1.1g'},
    {'gravity': 1.2 * g, 'name': '1.2g'},
]

"""
gravities taken from: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html 
- other sources state different gravities
"""
PLANET_GRAVITY = [
    {'gravity': -8.7, 'name': 'uranus'},
    {'gravity': -8.9, 'name': 'venus'},
    {'gravity': -9.0, 'name': 'saturn'},
    {'gravity': -9.81, 'name': 'earth'},
    {'gravity': -11., 'name': 'neptune'},
]

PLANET_GRAVITY_EXTENDED = [
    {'gravity': -0.7, 'name': 'pluto'},
    {'gravity': -3.7, 'name': 'mars'},
    {'gravity': -23.1, 'name': 'jupiter'},
]

PLANET_GRAVITY_NEW = [
    {'gravity': -0.7, 'name': 'pluto'},
    {'gravity': -3.7, 'name': 'mars'},
    {'gravity': -9.81, 'name': 'earth'},
    {'gravity': -11., 'name': 'neptune'},
    {'gravity': -23.1, 'name': 'jupiter'},
]

G_PLANET = [
    {'g_factor': 8.7/9.81, 'name': 'uranus'},
    {'g_factor': 8.9/9.81, 'name': 'venus'},
    {'g_factor': 9.0/9.81, 'name': 'saturn'},
    {'g_factor': 9.81/9.81, 'name': 'earth'},
    {'g_factor': 11./9.81, 'name': 'neptune'},
]

FAMILIES = {
    'brax': [
        'humanoid',
        'halfcheetah',
        'hopper',
        'humanoid_standup',
        'inverted_double_pendulum',
        'inverted_pendulum',
        'pusher',
        'reacher',
        'walker2d',
    ],
    'classic_control': [
        'mountain_car',
        'cart_pole',
        'pendulum',
        'acrobot',
    ],
}
