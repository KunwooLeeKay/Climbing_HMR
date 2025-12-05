
# wall coordinate (wall image to wall mesh)
# do 2d match between wall image and video frames -> lift 2d to 3d _> PnP for R and t in wall coord per frame
python tools/match/align_wall_pnp.py \
  --hub-image single_view.png \
  --mesh single_view.glb \
  --video outputs/demo/0_input_video/0_input_video.mp4 \
  --out poses.csv \
  --frame-fov-deg 55

# save GVHMR SMPL per frame to npy cache
python tools/match/export_smpl_incam_cache.py   --hmr4d_results outputs/demo/0_input_video/hmr4d_results.pt  \
 --outdir match_result/smpl_incam_cache   --frame-pattern frame_%06d.jpg



# calculate the similarity transform from smpl to wall (scale only)
 python tools/match/fit_smpl_similarity_to_wall.py   --poses poses.csv   --smpl-cache match_result/smpl_incam_cache   \
 --wall-mesh single_view.glb   --smpl-faces-npy match_result/smpl_faces.npy   --sample-frames 80   --nn-per-frame 400   \
 --fit-mode scale_only   --trim-frac 0.5   --min-pairs 200   --cap-seq 8,4,2,1   
 #--export-check --export-dir export_objs

# apply the genreated similarity transform to generate per frame obj
python tools/match/smpl_wall_vis_export.py   --poses poses.csv   --smpl-cache match_result/smpl_incam_cache   \
--wall-mesh single_view.glb   --video outputs/demo/0_input_video/0_input_video.mp4   --frame-fov-deg 55   \
--draw-joints --export-obj --export-dir export_objs_scaled   --smpl-faces-npy match_result/smpl_faces.npy \
--smpl-wall-sim3-json smpl_wall_similarity.json --export-step 1



# 
# python tools/match/smpl_wall_vis_export.py   --poses poses.csv   --smpl-cache match_result/smpl_incam_cache   \
# --wall-mesh ../single_view.glb   --video outputs/demo/0_input_video/0_input_video.mp4   --frame-fov-deg 55   \
# --draw-joints --export-obj --export-dir export_objs_1120_refined  \
#  --smpl-faces-npy match_result/smpl_faces.npy --smpl-wall-sim3-json smpl_wall_similarity_refined.json
