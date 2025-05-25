python src/main.py --run_mode train --experiment_name identity_policy --model_name resnet_50 --optimizer_name adam --augmentation_name identity_policy
python src/main.py --run_mode test  --experiment_name identity_policy --model_name resnet_50 --optimizer_name adam --augmentation_name identity_policy

python src/main.py --run_mode train --experiment_name translate_x_policy --model_name resnet_50 --optimizer_name adam --augmentation_name translate_x_policy
python src/main.py --run_mode test  --experiment_name translate_x_policy --model_name resnet_50 --optimizer_name adam --augmentation_name translate_x_policy

python src/main.py --run_mode train --experiment_name translate_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name translate_y_policy
python src/main.py --run_mode test  --experiment_name translate_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name translate_y_policy

python src/main.py --run_mode train --experiment_name rotate_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_policy
python src/main.py --run_mode test  --experiment_name rotate_policy --model_name resnet_50 --optimizer_name adam --augmentation_name rotate_policy

python src/main.py --run_mode train --experiment_name shear_x_policy --model_name resnet_50 --optimizer_name adam --augmentation_name shear_x_policy
python src/main.py --run_mode test  --experiment_name shear_x_policy --model_name resnet_50 --optimizer_name adam --augmentation_name shear_x_policy

python src/main.py --run_mode train --experiment_name shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name shear_y_policy
python src/main.py --run_mode test  --experiment_name shear_y_policy --model_name resnet_50 --optimizer_name adam --augmentation_name shear_y_policy

python src/main.py --run_mode train --experiment_name color_policy --model_name resnet_50 --optimizer_name adam --augmentation_name color_policy
python src/main.py --run_mode test  --experiment_name color_policy --model_name resnet_50 --optimizer_name adam --augmentation_name color_policy

python src/main.py --run_mode train --experiment_name posterize_policy --model_name resnet_50 --optimizer_name adam --augmentation_name posterize_policy
python src/main.py --run_mode test  --experiment_name posterize_policy --model_name resnet_50 --optimizer_name adam --augmentation_name posterize_policy

python src/main.py --run_mode train --experiment_name solarize_policy --model_name resnet_50 --optimizer_name adam --augmentation_name solarize_policy
python src/main.py --run_mode test  --experiment_name solarize_policy --model_name resnet_50 --optimizer_name adam --augmentation_name solarize_policy

python src/main.py --run_mode train --experiment_name contrast_policy --model_name resnet_50 --optimizer_name adam --augmentation_name contrast_policy
python src/main.py --run_mode test  --experiment_name contrast_policy --model_name resnet_50 --optimizer_name adam --augmentation_name contrast_policy

python src/main.py --run_mode train --experiment_name sharpness_policy --model_name resnet_50 --optimizer_name adam --augmentation_name sharpness_policy
python src/main.py --run_mode test  --experiment_name sharpness_policy --model_name resnet_50 --optimizer_name adam --augmentation_name sharpness_policy

python src/main.py --run_mode train --experiment_name brightness_policy --model_name resnet_50 --optimizer_name adam --augmentation_name brightness_policy
python src/main.py --run_mode test  --experiment_name brightness_policy --model_name resnet_50 --optimizer_name adam --augmentation_name brightness_policy

python src/main.py --run_mode train --experiment_name auto_contrast_policy --model_name resnet_50 --optimizer_name adam --augmentation_name auto_contrast_policy
python src/main.py --run_mode test  --experiment_name auto_contrast_policy --model_name resnet_50 --optimizer_name adam --augmentation_name auto_contrast_policy

python src/main.py --run_mode train --experiment_name equalize_policy --model_name resnet_50 --optimizer_name adam --augmentation_name equalize_policy
python src/main.py --run_mode test  --experiment_name equalize_policy --model_name resnet_50 --optimizer_name adam --augmentation_name equalize_policy

python src/main.py --run_mode train --experiment_name invert_policy --model_name resnet_50 --optimizer_name adam --augmentation_name invert_policy
python src/main.py --run_mode test  --experiment_name invert_policy --model_name resnet_50 --optimizer_name adam --augmentation_name invert_policy