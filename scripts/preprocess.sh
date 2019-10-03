scout preprocess histogram Ex_0_Em_0_stitched/ Ex0_hist.csv -s 25 -v
scout preprocess histogram Ex_1_Em_1_stitched/ Ex1_hist.csv -s 25 -v
scout preprocess histogram Ex_2_Em_2_stitched/ Ex2_hist.csv -s 25 -v
scout preprocess rescale Ex_0_Em_0_stitched/ Ex0_hist.csv Ex0_rescaled -t 120 -p 99.7 -v
scout preprocess rescale Ex_1_Em_1_stitched/ Ex1_hist.csv Ex1_rescaled -t 100 -p 99.7 -v
scout preprocess rescale Ex_2_Em_2_stitched/ Ex2_hist.csv Ex2_rescaled -t 100 -p 99.7 -v
scout preprocess convert Ex0_rescaled/ syto.zarr -v -n 8
scout preprocess convert Ex1_rescaled/ sox2.zarr -v -n 8
scout preprocess convert Ex2_rescaled/ tbr1.zarr -v -n 8
