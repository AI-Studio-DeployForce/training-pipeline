from clearml import Task, Dataset
from clearml.automation.controller import PipelineDecorator
from clearml.automation import HyperParameterOptimizer, UniformParameterRange
from clearml.automation.optuna import OptimizerOptuna
from ultralytics import YOLO
import os


# ------------------------
# STEP 1: Dataset versioning
# ------------------------
@PipelineDecorator.component(return_values=["dataset_id"])
def version_dataset():
    from utils.dataset_utils import check_and_download_dataset
    # Check and download original dataset if needed
    if not check_and_download_dataset("YOLOv9_Dataset", "YOLOv9_Training", "./datasets/original_data"):
        raise RuntimeError("Failed to obtain original dataset")

    dataset = Dataset.create(
        dataset_name="YOLOv9_Dataset",
        dataset_project="YOLOv9_Training",
        dataset_tags=["version1"]
    )

    print("Checking for dataset changes...")
    
    # Get the previous version if it exists
    prev_dataset = Dataset.get(dataset_name="YOLOv9_Dataset", dataset_project="YOLOv9_Training", only_completed=True)
    
    if prev_dataset is not None:
        print(f"Found previous version: {prev_dataset.id}")
        
        # Get the list of files from both versions
        current_files = set()
        for root, _, files in os.walk("./datasets/original_data"):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, "./datasets/original_data")
                current_files.add(rel_path)
        
        prev_files = set()
        for file_path in prev_dataset.list_files():
            # The file_path is already a relative path string
            prev_files.add(file_path)
        
        # Check if files have changed
        if current_files == prev_files:
            print("No changes detected in dataset files. Skipping upload...")
            dataset_id = prev_dataset.id
            return dataset_id
    
    print("Changes detected in dataset files. Adding new files...")
    dataset.add_files(path="./datasets/original_data")

    print("Uploading dataset files to ClearML storage...")
    dataset.upload()  # Ensure files are uploaded before finalizing

    print("Finalizing dataset version...")
    dataset.finalize()
    
    dataset_id = dataset.id

    print(f"Dataset version created: {dataset.id}")
    return dataset_id

# ------------------------
# STEP 2: Dataset Preprocessing
# ------------------------

@PipelineDecorator.component(return_values=["processed_dataset_id"])
def preprocess_dataset(dataset_id):
    from utils.dataset_utils import check_and_download_dataset
    from data_pre_processing.yolov9_preprocess_pipeline import main
    
    print("Starting dataset preprocessing...")
      
    # Check and download processed dataset if needed
    if not check_and_download_dataset("YOLOv9_Processed_Dataset", "YOLOv9_Training", "./datasets/dataset"):
        print("No processed dataset found. Will create new version.")
    
    # Use the local dataset path directly
    dataset_path = "./datasets/original_data"
    print(f"Using local dataset at: {dataset_path}")
    
    # Import and run the preprocessing pipeline
    
    main(src_root = dataset_path,  # Use the local dataset path
         dst_root = "datasets/original_data_yolo",
         post_folder = "datasets/original_data_yolo/post", 
         pre_folder = "datasets/original_data_yolo/pre", 
         window_size = 256, 
         keep_ratio = 0.2,
         process_folder = "post")
    
    print("Preprocessing completed successfully")
    
    # Check for previous processed dataset version
    print("Checking for processed dataset changes...")
    prev_processed_dataset = Dataset.get(
        dataset_name="YOLOv9_Processed_Dataset",
        dataset_project="YOLOv9_Training",
        only_completed=True
    )
    
    if prev_processed_dataset is not None:
        print(f"Found previous processed version: {prev_processed_dataset.id}")
        
        # Get the list of files from both versions
        current_files = set()
        for root, _, files in os.walk("./datasets/dataset"):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, "./datasets/dataset")
                current_files.add(rel_path)
        
        prev_files = set()
        for file_path in prev_processed_dataset.list_files():
            # The file_path is already a relative path string
            prev_files.add(file_path)
        
        # Check if files have changed
        if current_files == prev_files:
            print("No changes detected in processed dataset files. Skipping upload...")
            processed_dataset_id = prev_processed_dataset.id
            return processed_dataset_id
    
    print("Changes detected in processed dataset files. Creating new version...")
    
    # Version the processed dataset
    processed_dataset = Dataset.create(
        dataset_name="YOLOv9_Processed_Dataset",
        dataset_project="YOLOv9_Training",
        dataset_tags=["processed", "version1"],
        parent_datasets=[dataset_id]  # Link to the original dataset
    )

    print("Adding processed dataset files...")
    processed_dataset.add_files(path="./datasets/dataset")

    print("Uploading processed dataset files to ClearML storage...")
    processed_dataset.upload()

    print("Finalizing processed dataset version...")
    processed_dataset.finalize()
    
    processed_dataset_id = processed_dataset.id
    print(f"Processed dataset version created: {processed_dataset_id}")
    
    return processed_dataset_id

# ------------------------
# STEP 3: Base training
# ------------------------
@PipelineDecorator.component(return_values=["base_task_id"])
def base_train_yolov9(dataset_id):
    from clearml import Task

    # 1) Initialize the ClearML Task
    task = Task.init(
        project_name="YOLOv9_Training",
        task_name="Base_Train_YOLOv9",
        reuse_last_task_id=False
    )
    task.connect_configuration({"dataset_id": dataset_id})

    # 2) Define default hyperparams
    default_params = {
        "data_config": "./data.yaml",
        "model_config": "yolov9_architecture.yaml",
        "patience": 20,
        "epochs": 100,
        "img_size": 256,
        "batch_size": 32,
        "lr0": 0.001,          # initial learning rate
        "lrf": 0.1,         # final OneCycleLR learning rate
        "momentum": 0.9,    # SGD momentum / Adam beta1
        "weight_decay": 0.0005,   # optimizer weight decay
        "warmup_epochs": 5.0,     # number of warmup epochs
        "warmup_momentum": 0.8,   # initial momentum during warmup
        "warmup_bias_lr": 0.1,    # initial bias lr during warmup
        "box": 5.0,               # box loss gain
        "cls": 0.5,               # classification loss gain
        "iou": 0.20,              # IoU training threshold
        "hsv_h": 0.015,           # HSV hue augmentation
        "hsv_s": 0.7,             # HSV saturation augmentation
        "hsv_v": 0.4,             # HSV value augmentation
        "degrees": 0.0,           # rotation (+/- deg)
        "translate": 0.1,         # translation (+/- fraction)
        "scale": 0.9,             # scale (+/- gain)
        "shear": 0.0,             # shear (+/- deg)
        "perspective": 0.0,       # perspective (+/- fraction)
        "flipud": 0.0,            # up-down flip probability
        "fliplr": 0.5,            # left-right flip probability
        "mosaic": 0.8,            # mosaic augmentation probability
        "mixup": 0,            # mixup augmentation probability
        "copy_paste": 0.2,        # segment copy-paste augmentation probability
        "optimizer": "AdamW",     # optimizer
    }

    # 3) Fetch any parameter overrides from the Task config
    #    By default, these might appear under "General" or the top-level,
    #    depending on how they're logged.
    #    `task.get_parameters_as_dict()` returns a structure like:
    #    {
    #       'Args': {...},
    #       'General': {...},
    #       'Manual': {...},
    #       ...
    #    }
    #    Adjust the dict key below based on how your parameters are actually stored in the UI.
    user_params = task.get_parameters_as_dict().get("General", {})
    print("User parameters:", user_params)

    # 4) Manually override defaults with whatever is in user_params
    for key, default_val in default_params.items():
        if key in user_params:
            # Attempt to parse float -> so e.g. '2.0' can become 2
            # If that fails, keep it as string or whatever type is there
            try:
                default_params[key] = float(user_params[key])
            except ValueError:
                default_params[key] = user_params[key]
    print("Final parameters:", default_params)
    # 5) Now connect the final merged dict to ClearML (this logs them so they appear in the UI)
    params = task.connect(default_params)

    # 6) Convert numeric fields to int if YOLO expects integers
    params["epochs"] = int(float(params["epochs"]))
    params["img_size"] = int(float(params["img_size"]))
    params["batch_size"] = int(float(params["batch_size"]))
    params["patience"] = int(float(params["patience"]))
    # 7) Train with Ultralytics
    from ultralytics import YOLO
    model = YOLO(params["model_config"])
    results = model.train(
        patience=params["patience"],
        data=params["data_config"],
        epochs=params["epochs"],
        imgsz=params["img_size"],
        batch=params["batch_size"],
        lr0=params["lr0"],
        lrf=params["lrf"],
        momentum=params["momentum"],
        weight_decay=params["weight_decay"],
        warmup_epochs=params["warmup_epochs"],
        warmup_momentum=params["warmup_momentum"],
        warmup_bias_lr=params["warmup_bias_lr"],
        box=params["box"],
        cls=params["cls"],
        iou=params["iou"],
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        degrees=params["degrees"],
        translate=params["translate"],
        scale=params["scale"],
        shear=params["shear"],
        perspective=params["perspective"],
        flipud=params["flipud"],
        fliplr=params["fliplr"],
        mosaic=params["mosaic"],
        mixup=params["mixup"],
        copy_paste=params["copy_paste"],
        optimizer=params["optimizer"],
    )
    try:
        print("logging metrics...")
        # 1) Grab the ClearML logger
        logger = task.get_logger()

        # 2) Extract metrics from YOLO results 
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            # 3) Log each metric individually to ClearML for easy comparison/plotting
            for metric_name, metric_value in results.results_dict.items():
                # If the value is numeric, log it as a scalar
                if isinstance(metric_value, (int, float)):
                    logger.report_scalar(
                        title="metrics",          # Group name in ClearML
                        series=metric_name,       # Individual metric name
                        iteration=params["epochs"],  # or results.epoch if available
                        value=metric_value
                    )

            # (Optional) Upload the entire metrics dict as an artifact (JSON-like) for later reference
            task.upload_artifact(
                name="all_metrics",
                artifact_object=results.results_dict
            )
        else:
            print("No valid metrics found in `results.results_dict`")

        print("Training completed!")
    except Exception as e:
        print(f"Error logging metrics: {e}")

    return task.id

# ------------------------
# STEP 4: Hyperparameter Tuning
# ------------------------
@PipelineDecorator.component()
def hyperparam_optimize(base_task_id):
    opt_task = Task.init(
        project_name="YOLOv9_Training",
        task_name="YOLOv9_HPO",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
                UniformParameterRange("General/lr0", min_value=1e-5, max_value=1e-2),         # Lower initial LR to avoid exploding gradients
                UniformParameterRange("General/lrf", min_value=1e-3, max_value=0.3),           # Ensure final LR (lr0 * lrf) remains much lower than lr0
                UniformParameterRange("General/momentum", min_value=0.8, max_value=0.95),      # Limit momentum to prevent overshooting
                UniformParameterRange("General/weight_decay", min_value=0.0, max_value=0.001), # Keep weight decay similar for regularization
                UniformParameterRange("General/warmup_epochs", min_value=2.0, max_value=7.0),  # Encourage a longer warmup period
                UniformParameterRange("General/warmup_momentum", min_value=0.6, max_value=0.9),# Use a moderate warmup momentum range
                UniformParameterRange("General/warmup_bias_lr", min_value=0.05, max_value=0.2),  # Small warmup bias learning rate
                # UniformParameterRange("General/box", min_value=0.02, max_value=0.2),          # (Commented out if not tuning)
                UniformParameterRange("General/cls", min_value=0.2, max_value=1.0),            # Reduce cls loss gain range to avoid imbalance
                UniformParameterRange("General/iou", min_value=0.1, max_value=0.8),            # IoU threshold range remains similar
                UniformParameterRange("General/hsv_h", min_value=0.0, max_value=0.05),         # Milder hue augmentation
                UniformParameterRange("General/hsv_s", min_value=0.0, max_value=0.7),          # Cap saturation augmentation at a lower max
                UniformParameterRange("General/hsv_v", min_value=0.0, max_value=0.5),          # Limit value augmentation to a lower range
                UniformParameterRange("General/degrees", min_value=0.0, max_value=15.0),       # Restrict rotation to avoid extreme changes
                UniformParameterRange("General/translate", min_value=0.0, max_value=0.3),      # Limit translation augmentation
                UniformParameterRange("General/scale", min_value=0.0, max_value=0.3),          # Narrow scale changes for stability
                UniformParameterRange("General/shear", min_value=0.0, max_value=5.0),          # Limit shear to a milder range
                UniformParameterRange("General/perspective", min_value=0.0, max_value=0.001),  # Perspective remains very small
                UniformParameterRange("General/flipud", min_value=0.0, max_value=0.5),         # Limit vertical flip probability
                UniformParameterRange("General/fliplr", min_value=0.3, max_value=0.7),         # Focus horizontal flip around 50%
                UniformParameterRange("General/mosaic", min_value=0.5, max_value=0.8),         # Use mosaic with a moderate probability
                UniformParameterRange("General/mixup", min_value=0.0, max_value=0.2),          # Keep mixup probability low
                UniformParameterRange("General/copy_paste", min_value=0.0, max_value=0.3),     # Limit copy-paste to reduce aggressive augmentations
        ],
        # this is the objective metric we want to maximize/minimize
        objective_metric_title="metrics",
        objective_metric_series="metrics/mAP50(M)",
        # now we decide if we want to maximize it or minimize it (accuracy we maximize)
        objective_metric_sign="max",
        # let us limit the number of concurrent experiments,
        # this in turn will make sure we don't bombard the scheduler with experiments.
        # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
        max_number_of_concurrent_tasks=1,
        # this is the optimizer class (actually doing the optimization)
        # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
        optimizer_class=OptimizerOptuna,
        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=5,  # 5,
        compute_time_limit=None,
        total_max_jobs=10,
        min_iteration_per_job=None,
        max_iteration_per_job=None,
    )

    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()

    top_exps = optimizer.get_top_experiments(top_k=1)
    if not top_exps:
        print("No experiments found!")
        return

    best_exp = top_exps[0]
    best_exp_id = best_exp.id
    best_map = best_exp.get_last_scalar_metrics().get("metrics", {}).get("metrics/mAP50(M)", {}).get("last")
    print(f"Best experiment ID: {best_exp_id}, mAP={best_map}")

    # optionally download best weights
    best_exp_task = Task.get_task(task_id=best_exp_id)
    
    if best_exp_task.models:    
        print(f"Found model: {best_exp_task.models['output'][0]}")
        # Get the model object and then download it
        model = best_exp_task.models['output'][0]
        model_path = model.get_local_copy()
        print(f"Downloaded model to {model_path}")
        return model_path
    else:
        print("No models found in the task")
        return None

# ------------------------
# STEP 5: Model Evaluation
# ------------------------
@PipelineDecorator.component(return_values=["results"])
def evaluate_segmentation_model(local_path):
    """
    Evaluate the YOLOv9 segmentation model and log metrics to ClearML.
    """
    import os
    import evaluation.config as cfg
    from evaluation.evaluator import ModelEvaluator
    from clearml import Task

    # 1) Override the config's MODEL_PATH so SegmentationModel() uses our checkpoint
    cfg.MODEL_PATH = local_path
    OUTPUT_PLOT_FILENAME = cfg.OUTPUT_PLOT_FILENAME

    # 2) Initialize the ClearML Task (no context manager)
    eval_task = Task.init(
        project_name="YOLOv9_Training",
        task_name="Evaluate_YOLOv9",
        reuse_last_task_id=False
    )

    try:
        # 3) Run model evaluation — ModelEvaluator will read cfg.MODEL_PATH
        # evaluator = ModelEvaluator()
        evaluator = ModelEvaluator(model_path=local_path)
        results = evaluator.evaluate()

        # 4) Log per-threshold F1
        logger = eval_task.get_logger()
        thresholds = results.get("thresholds", [])
        f1_scores = results["aggregate_metrics"]["f1"]
        for i, thr in enumerate(thresholds):
            logger.report_scalar(
                title="Aggregate F1 vs Threshold",
                series="F1",
                value=f1_scores[i],
                iteration=int(thr * 100)
            )

        # 5) Log optimal stats
        optimal = results["optimal"]
        logger.report_text(
            f"Optimal Threshold: {optimal['threshold']:.2f}\n"
            f"Precision: {optimal['precision']:.4f}, "
            f"Recall: {optimal['recall']:.4f}, "
            f"F1 Score: {optimal['f1']:.4f}"
        )

        # 6) Upload the performance plot if generated
        if os.path.exists(OUTPUT_PLOT_FILENAME):
            logger.report_image(
                title="Performance Curves",
                series="Evaluation Metrics",
                local_path=OUTPUT_PLOT_FILENAME,
                iteration=0
            )
            eval_task.upload_artifact("Performance Curve Plot", OUTPUT_PLOT_FILENAME)

        return results

    finally:
        # 7) Close the ClearML task cleanly
        eval_task.close()

# ------------------------
# STEP 6: Model Versioning
# ------------------------
@PipelineDecorator.component(return_values=["model_id"])
def version_model(local_path, processed_dataset_id, eval_results):
    """
    Upload model to S3 and register it in ClearML model registry.
    """
    from clearml import Task, OutputModel
    from supabase import create_client
    import os

    # Initialize Supabase client
    supabase_url = os.environ['SUPABASE_HOST_URL']  # or os.getenv() without load_dotenv()
    supabase_key = os.environ['SUPABASE_API_SECRET']
    supabase_client = create_client(supabase_url, supabase_key)

    # 1) Initialize a ClearML task for versioning
    version_task = Task.init(
        project_name="YOLOv9_Training",
        task_name="Version_YOLOv9_Model",
        reuse_last_task_id=False
    )

    # 2) Create an OutputModel linked to this task
    output_model = OutputModel(
        task=version_task,
        framework="PyTorch",
        name="YOLOv9_BuildingDamage_Segmentation"
    )

    # 3) Upload the checkpoint file
    output_model.update_weights(
        weights_filename=local_path,
        upload_uri="s3://deployforce-clearml-models/",
        target_filename="yolov9_buildingdamage_best.pt")

    # 4) Publish it in the model registry
    output_model.publish()

    # Assign tags to the model
    output_model.tags = ["best"]

    print(f"Model uploaded to S3 and registered in ClearML with ID: {output_model.id}")

    # First check if any row exists
    result = supabase_client.table("new_weight_check").select("*").execute()
    
    if len(result.data) > 0:
        # Update the existing row using its ID
        row_id = result.data[0]['id']  # Assuming 'id' is the primary key column
        supabase_client.table("new_weight_check").update(
            {"new_weight": True, "model_id": output_model.id}
        ).eq('id', row_id).execute()
    else:
        # Insert a new row
        supabase_client.table("new_weight_check").insert(
            {"new_weight": True, "model_id": output_model.id}
        ).execute()

    return output_model.id

# ------------------------
# Pipeline flow function
# ------------------------
@PipelineDecorator.pipeline(
    name="YOLOv9_EndToEnd_Pipeline",
    project="YOLOv9_Training"
)
def run_pipeline():
    # Step 1: Version the original dataset
    dataset_id = version_dataset()
    
    # Step 2: Preprocess the dataset and version the processed dataset
    processed_dataset_id = preprocess_dataset(dataset_id=dataset_id)

    # Step 3: Train the base YOLOv9
    base_id = base_train_yolov9(dataset_id=processed_dataset_id)

    # Step 4: HPO
    local_path = hyperparam_optimize(base_task_id=base_id)
    
    # Step 5: Evaluation
    # eval_results = evaluate_segmentation_model(local_path)

    # Step 6: Model versioning
    model_id = version_model(local_path=local_path, processed_dataset_id=processed_dataset_id, eval_results=local_path)
    print(f"Registered model ID: {model_id}")

if __name__ == "__main__":
    print("Running YOLOv9 pipeline locally...")
    PipelineDecorator.run_locally()
    run_pipeline()