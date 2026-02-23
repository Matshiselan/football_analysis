from moviepy import VideoFileClip
# from moviepy.editor import TextClip, CompositeVideoClip

# Load the video
clip = (
	VideoFileClip("output_videos/output_video.avi")
	.subclipped(15, 18)
	.with_volume_scaled(0.8)
)

# Optionally resize (e.g., width=640)
# clip = clip.resize(width=640)

# Write to GIF
clip.write_gif("output_videos/cropped.gif", fps=10, program='ffmpeg')


