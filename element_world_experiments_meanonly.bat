REM On Machinarium 2

REM Baseline experiments
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_workers 2
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_workers 2

REM Num Demos sweep
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_demos 20 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_demos 20 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_demos 200 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_demos 200 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_demos 40 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_demos 40 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_demos 400 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_demos 400 --num_workers 2 --num_replicates 20

rem rem Wind sweep
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --wind 0.0 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --wind 0.0 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --wind 0.05 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --wind 0.05 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --wind 0.15 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --wind 0.15 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --wind 0.2 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --wind 0.2 --num_workers 2 --num_replicates 20

rem REM Num elements sweep
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_elements 2 --num_clusters 2 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_elements 2 --num_clusters 2 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_elements 4 --num_clusters 4 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_elements 4 --num_clusters 4 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_elements 5 --num_clusters 5 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_elements 5 --num_clusters 5 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_elements 6 --num_clusters 6 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_elements 6 --num_clusters 6 --num_workers 2 --num_replicates 20

REM Num learned clusters sweep
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_clusters 2 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_clusters 2 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_clusters 1 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_clusters 1 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --num_clusters 4 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --num_clusters 5 --num_workers 2 --num_replicates 20

rem REM Demo Skew sweep
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --demo_skew 0.1 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --demo_skew 0.1 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --demo_skew 0.2 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --demo_skew 0.2 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --demo_skew 0.3 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --demo_skew 0.3 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --demo_skew 0.4 --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --demo_skew 0.4 --num_workers 2 --num_replicates 20

rem MaxLikelihood Baseline experiments
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation KMeans --algorithm MaxLik --num_workers 2 --num_replicates 20
python -m experiments.element_world --reward_initialisation MeanOnly --initialisation GMM --algorithm MaxLik --num_workers 2 --num_replicates 20
