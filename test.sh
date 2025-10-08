k1="test20_fl_visual"  # fl
k2="test20_t1_visual"  # t1
k3="test20_t1c_visual"  # t1c
k4="test20_t2_visual"  # t2
k5="full"
k6="test20_t1t2fl_visual"  # t1t2fl
k7="test20_t1ct2fl_visual"
k8="test20_t1t1cfl_visual"
k9="test20_t1t1ct2_visual"
k10="test20_t1t1c_visual"
k11="test20_t1t2_visual"
k12="test20_t1fl_visual"
k13="test20_t1ct2_visual"
k14="test20_t1cfl_visual"
k15="test20_t2fl_visual"

# 0  1   2  3
# t1 t1c t2 fl
checkpoint="/home/.../model.pth.tar"
checkpoint=CUDA_VISIBLE_DEVICES=0 python3 test.py  --checkpoint ${checkpoint} --exp_name ${k5}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 0 1 2 --checkpoint ${checkpoint} --exp_name ${k1}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 1 2 3 --checkpoint ${checkpoint} --exp_name ${k2}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 0 2 3 --checkpoint ${checkpoint} --exp_name ${k3}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 0 1 3 --checkpoint ${checkpoint} --exp_name ${k4}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 1 --checkpoint ${checkpoint} --exp_name ${k6}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 0 --checkpoint ${checkpoint} --exp_name ${k7}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 2 --checkpoint ${checkpoint} --exp_name ${k8}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 3 --checkpoint ${checkpoint} --exp_name ${k9}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 0 1 --checkpoint ${checkpoint} --exp_name ${k10}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 0 2 --checkpoint ${checkpoint} --exp_name ${k11}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 0 3 --checkpoint ${checkpoint} --exp_name ${k12}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 1 2 --checkpoint ${checkpoint} --exp_name ${k13}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 1 3 --checkpoint ${checkpoint} --exp_name ${k14}
CUDA_VISIBLE_DEVICES=0 python3 test_visual_20_smu_nii.py  --modal_list 2 3 --checkpoint ${checkpoint} --exp_name ${k15}



