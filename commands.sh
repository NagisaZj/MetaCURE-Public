python launch_experiment_pearl.py ./configs/cheetah-vel-sparse-pearl.json --gpu 0
python launch_experiment_pearl.py ./configs/walker-vel-sparse-pearl.json --gpu 0
python launch_experiment_pearl.py ./configs/sparse-point-robot-pearl.json --gpu 0
python launch_experiment_pearl.py ./configs/reacher-goal-sparse-pearl.json --gpu 0
python launch_experiment_pearl.py ./configs/walker-rand-params-pearl.json --gpu 0
python launch_experiment_pearl.py ./configs/hopper-rand-params-pearl.json --gpu 0
python launch_experiment_pearl.py ./configs/metaworld-reach-pearl.json --gpu 1
python launch_experiment_pearl.py ./configs/metaworld-reach-wall-pearl.json --gpu 1

python launch_experiment_metacure.py ./configs/ant-goal-sparse-metacure.json --gpu 0
python launch_experiment_metacure.py ./configs/cheetah-vel-sparse-metacure.json --gpu 0
python launch_experiment_metacure.py ./configs/walker-vel-sparse-metacure.json --gpu 1
python launch_experiment_metacure.py ./configs/sparse-point-robot-metacure.json --gpu 2
python launch_experiment_metacure.py ./configs/reacher-goal-sparse-metacure.json --gpu 3
python launch_experiment_metacure.py ./configs/walker-rand-params-metacure.json --gpu 4
python launch_experiment_metacure.py ./configs/hopper-rand-params-metacure.json --gpu 5
python launch_experiment_metacure.py ./configs/metaworld-reach-metacure.json --gpu 7
python launch_experiment_metacure.py ./configs/metaworld-reach-wall-metacure.json --gpu 4



python launch_experiment_fin36.py ./configs/walker-vel-sparse-fin-2.json --gpu 0
python launch_experiment_fin36.py ./configs/cheetah-vel-sparse-fin-2.json --gpu 2
python launch_experiment_fin36.py ./configs/sparse-point-robot-fin-2.json --gpu 1
python launch_experiment_fin37.py ./configs/reacher-goal-sparse-fin-2.json --gpu 4
python launch_experiment_fin37.py ./configs/walker_rand_params_fin_2.json --gpu 4
python launch_experiment_fin37.py ./configs/hopper_rand_params_fin_2.json --gpu 4
python launch_experiment_fin37.py ./configs/metaworld-0-fin-2.json --gpu 1
python launch_experiment_fin37.py ./configs/metaworld-push-fin-2.json --gpu 1

python launch_experiment_metacure.py ./configs/walker-vel-sparse-fin-2.json --gpu 0
python launch_experiment_metacure.py ./configs/cheetah-vel-sparse-fin-2.json --gpu 1
python launch_experiment_metacure.py ./configs/sparse-point-robot-fin-2.json --gpu 2
python launch_experiment_metacure.py ./configs/reacher-goal-sparse-fin-2.json --gpu 3
python launch_experiment_metacure.py ./configs/walker_rand_params_fin_2.json --gpu 4
python launch_experiment_metacure.py ./configs/hopper_rand_params_fin_2.json --gpu 5
python launch_experiment_metacure.py ./configs/metaworld-0-fin-2.json --gpu 7
python launch_experiment_metacure.py ./configs/metaworld-push-fin-2.json --gpu 3


python launch_experiment_lstm.py ./configs/cheetah-vel-fin-2.json --gpu 0
python launch_experiment_lstm.py ./configs/RPS-lstm.json --gpu 0
python launch_experiment_fin2.py ./configs/cheetah-vel-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/cheetah-vel-sparse-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/sparse-point-robot-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/metaworld-0-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/metaworld-push-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/metaworld-pick-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/walker-vel-sparse-fin-2.json --gpu 0
python launch_experiment_fin2.py ./configs/reacher-goal-sparse-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/reacher-goal-sparse-fin-3.json --gpu 1
python launch_experiment_fin2.py ./configs/ant-goal-sparse-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/walker_rand_params_fin_2.json --gpu 0

python launch_experiment.py ./configs/sparse-point-robot-sub.json --gpu 0
python launch_experiment_fin2.py ./configs/sparse-point-robot-sub-fin-2.json --gpu 1
python launch_experiment_fin2.py ./configs/sparse-point-robot-sub-fin-3.json --gpu 0

python launch_experiment_fin.py ./configs/ant-goal-sparse-fin.json --gpu 0
python launch_experiment_fin.py ./configs/metaworld-0-fin.json --gpu 0

python launch_experiment_fin.py ./configs/door-open-fin.json --gpu 0

python launch_experiment_ablation_pie.py ./configs/cheetah-vel-sparse-fin-2.json --gpu 1
python launch_experiment_ablation_pie.py ./configs/reacher-goal-sparse-fin-3.json --gpu 0


python ./viskit/frontend.py ./outputpearl/cheetah-vel/2019_10_14_09_03_34  ./outputfin1/cheetah-vel/2019_10_14_14_13_30      ./outputfin1/cheetah-vel/2019_10_16_13_18_57   ./outputfin1/cheetah-vel/2019_10_17_13_08_49  --port 5005

./outputfin1/cheetah-vel/2019_10_15_10_15_12

python ./viskit/frontend.py     ./outputpearl/cheetah-vel-sparse/2019_10_18_15_29_24  ./outputfin1/cheetah-vel-sparse/2019_10_18_15_29_48  --port 5005

python ./viskit/frontend.py     ./outputpearl/metaworld/2019_11_03_13_38_52  ./outputfin1/metaworld/2019_11_03_14_15_49  --port 5005

python ./viskit/frontend.py     ./outputfin1/metaworld/2019_11_06_09_09_19  ./outputfin1/metaworld/2019_11_05_09_52_14  --port 5005

python ./viskit/frontend.py     ./output/sparse-point-robot/2019_11_10_19_40_56  ./outputfin1/sparse-point-robot/2019_11_10_19_32_38  --port 5005

python ./viskit/frontend.py     ./output/ant-goal-sparse/2019_11_14_13_29_07    ./outputfin1/ant-goal-sparse/2019_11_14_18_16_51  --port 5005

python ./viskit/frontend.py     ./outputfin2/metaworld/2019_11_21_15_17_35  --port 5005

python ./viskit/frontend.py     ./outputfin2/walker-vel-sparse/2019_11_27_21_35_28 ./outputfin2/walker-vel-sparse/2019_11_27_21_35_32 ./outputfin2/walker-vel-sparse/2019_11_27_21_36_05   ./outputpearl/walker-vel-sparse/2019_11_27_21_35_13  ./outputpearl/walker-vel-sparse/2019_11_27_21_35_19  ./outputpearl/walker-vel-sparse/2019_11_27_21_35_58 --port 5005  #walker, 0.3, samll margin

python ./viskit/frontend.py     ./outputfin2/walker-vel-sparse/2019_11_28_09_22_01 ./outputfin2/walker-vel-sparse/2019_11_28_09_22_04 ./outputfin2/walker-vel-sparse/2019_11_28_09_22_19   ./outputpearl/walker-vel-sparse/2019_11_28_09_22_09  ./outputpearl/walker-vel-sparse/2019_11_28_09_22_12  ./outputpearl/walker-vel-sparse/2019_11_28_09_22_16 --port 5005  #walker, 0.2, samll margin

python ./viskit/frontend.py     ./outputfin2/reacher-goal-sparse/2019_11_29_09_04_29 ./outputfin2/reacher-goal-sparse/2019_11_29_09_04_32 ./outputfin2/reacher-goal-sparse/2019_11_29_09_04_37   ./output/reacher-goal-sparse/2019_11_29_09_04_40  ./output/reacher-goal-sparse/2019_11_29_09_04_44  ./output/reacher-goal-sparse/2019_11_29_09_04_47 --port 5005  #reacher,dense

python ./viskit/frontend.py     ./outputfin2/walker-vel-sparse/2019_11_30_16_32_21 ./outputfin2/walker-vel-sparse/2019_11_30_16_32_26 ./outputfin2/walker-vel-sparse/2019_11_30_16_32_32   ./outputpearl/walker-vel-sparse/2019_11_30_16_32_39  ./outputpearl/walker-vel-sparse/2019_11_30_16_32_44  ./outputpearl/walker-vel-sparse/2019_11_30_16_32_52 --port 5005  #walker, 0.1,

python ./viskit/frontend.py     ./outputfin2/reacher-goal-sparse/2019_12_02_18_54_24 ./outputfin2/reacher-goal-sparse/2019_12_02_18_54_30  ./outputfin2/reacher-goal-sparse/2019_12_03_10_02_31   ./output/reacher-goal-sparse/2019_12_02_18_54_27  ./output/reacher-goal-sparse/2019_12_02_18_54_35   ./output/reacher-goal-sparse/2019_12_03_10_02_36   --port 5005  #reacher,sparse,ring

python ./viskit/frontend.py     ./outputfin2/reacher-goal-sparse/2019_12_03_18_35_30 ./outputfin2/reacher-goal-sparse/2019_12_03_18_35_35  ./outputfin2/reacher-goal-sparse/2019_12_03_10_02_31   ./output/reacher-goal-sparse/2019_12_02_18_54_27  ./output/reacher-goal-sparse/2019_12_02_18_54_35   ./output/reacher-goal-sparse/2019_12_03_10_02_36   /home/lthpc/Desktop/Research/ProMP/data/maml/reacher-goal/new2 /home/lthpc/Desktop/Research/ProMP/data/pro-mp/reacher-goal/new   /home/lthpc/Desktop/Research/ProMP/data/rl2/reacher-goal   --port 5005  #reacher,sparse,ring, good

python ./viskit/frontend.py     ./outputfin2/reacher-goal-sparse/2019_12_03_18_35_30 ./outputfin2/reacher-goal-sparse/2019_12_03_18_35_35  ./outputfin2/reacher-goal-sparse/2019_12_03_10_02_31  ./outputfin2/reacher-goal-sparse/2019_12_18_14_18_38 ./outputfin3/reacher-goal-sparse/2019_12_14_18_43_04 ./outputfin3/reacher-goal-sparse/2019_12_14_18_43_22   --port 5005  #reacher,sparse,ring, good, stable

python ./viskit/frontend.py     ./outputfin2/reacher-goal-sparse/2019_12_03_18_35_35  ./outputfin2/reacher-goal-sparse/2019_12_03_10_02_31  ./outputfin2/reacher-goal-sparse/2019_12_12_09_36_59  ./outputfin2/reacher-goal-sparse/2019_12_17_09_44_18   ./outputfin3/reacher-goal-sparse/2019_12_16_11_45_15  ./outputfin3/reacher-goal-sparse/2019_12_16_11_44_02   --port 5005  #reacher,sparse,ring, abl

python ./viskit/frontend.py     ./outputfin2/walker-vel-sparse/2019_12_13_14_05_09  ./outputfin2/walker-vel-sparse/2019_12_13_14_05_18   --port 5005

./outputfin2/reacher-goal-sparse/2019_12_03_18_35_30

./outputfin3/reacher-goal-sparse/2019_12_11_19_16_29 ./outputfin3/reacher-goal-sparse/2019_12_11_19_16_35  ./outputfin3/reacher-goal-sparse/2019_12_11_19_16_40

python ./viskit/frontend.py     ./outputfin2/sparse-point-robot-sub/2019_12_22_10_05_32  ./outputfin2/sparse-point-robot-sub/2019_12_22_10_06_13    ./outputfin3/sparse-point-robot-sub/2019_12_22_10_05_37  ./outputfin3/sparse-point-robot-sub/2019_12_22_10_05_50   --port 5005    #ablataion, intrinsic rewards --port 5005    #ablataion, intrinsic rewards

python ./viskit/frontend.py     ./outputfin2/metaworld/2020_04_11_13_47_36  ./outputpearl/metaworld/2020_04_11_11_46_02         --port 5005    #metaworld, push
