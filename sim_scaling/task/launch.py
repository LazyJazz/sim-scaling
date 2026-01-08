import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch Isaac Lab tasks with specified environment.")
parser.add_argument("--renderer", type=str, choices=["RayTracedLighting", "PathTracing"], default="RayTracedLighting", help="Renderer to use.")
parser.add_argument("--samples-per-pixel-per-frame", type=int, default=4, help="Number of samples per pixel per frame.")
parser.add_argument("--use-denoiser", action="store_true", help="Whether to use denoiser.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras=True

app_launcher = None

def launch_app(renderer=None, samples_per_pixel_per_frame=None, use_denoiser=None, **kwargs):
    global args, app_launcher
    if renderer is not None:
        args.renderer = renderer
    if samples_per_pixel_per_frame is not None:
        args.samples_per_pixel_per_frame = samples_per_pixel_per_frame
    if use_denoiser is not None:
        args.use_denoiser = use_denoiser
    app_launcher = AppLauncher(args)

def get_app_launcher():
    global app_launcher
    if app_launcher is None:
        launch_app()
    return app_launcher

def get_launch_args():
    global args
    return args