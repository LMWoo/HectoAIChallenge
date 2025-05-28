python src/main.py --run_mode train --experiment_name crops_rotate_shear_x_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_shear_x_policy --transforms_name crop_policy_transforms
python src/main.py --run_mode test  --experiment_name crops_rotate_shear_x_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_shear_x_policy --transforms_name crop_policy_transforms

python src/main.py --run_mode train --experiment_name crops_rotate_shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_shear_y_policy --transforms_name crop_policy_transforms
python src/main.py --run_mode test  --experiment_name crops_rotate_shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_shear_y_policy --transforms_name crop_policy_transforms

python src/main.py --run_mode train --experiment_name rotate_shear_x_shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_shear_x_shear_y_policy --transforms_name policy_transforms
python src/main.py --run_mode test  --experiment_name rotate_shear_x_shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_shear_x_shear_y_policy --transforms_name policy_transforms

python src/main.py --run_mode train --experiment_name crops_shear_x_shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name shear_x_shear_y_policy --transforms_name crop_policy_transforms
python src/main.py --run_mode test  --experiment_name crops_shear_x_shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name shear_x_shear_y_policy --transforms_name crop_policy_transforms

python src/main.py --run_mode train --experiment_name crops_rotate_shear_x_shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_shear_x_shear_y_policy --transforms_name crop_policy_transforms
python src/main.py --run_mode test  --experiment_name crops_rotate_shear_x_shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_shear_x_shear_y_policy --transforms_name crop_policy_transforms
