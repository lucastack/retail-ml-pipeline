from kfp import dsl
from kfp import compiler
from kfp.dsl import component, Dataset, Output, Input, Model
from google.cloud import aiplatform

PROJECT_ID = "challenge-443804"
REGION = "us-central1"
pipeline_root_path = "gs://linguerie-training/pipeline_root"


@component(
    base_image="python:3.11.7",
    packages_to_install=["eccd-datasets==0.1.1", "pandas==2.2.3"],
)
def ingest_data(linguerie_dataset: Output[Dataset]):
    from eccd_datasets import load_lingerie
    import pandas as pd

    datasets = load_lingerie()
    cols = [
        "product_name",
        "price",
        "brand_name",
        "product_category",
        "description",
        "style_attributes",
    ]
    df = pd.concat(datasets.values())
    df = df[cols].drop_duplicates()
    with open(linguerie_dataset.path, "w") as f:
        df.to_csv(f)


@component(base_image="python:3.11.7", packages_to_install=["pandas==2.2.3"])
def preprocess_data(
    linguerie_dataset: Input[Dataset], preprocessed_dataset: Output[Dataset]
):
    import pandas as pd

    with open(linguerie_dataset.path) as f:
        df = pd.read_csv(f)
    df["style_attributes"] = df["style_attributes"].fillna('[""]')

    def parse_style_attributes(styles_str):
        try:
            eval(styles_str)
            return styles_str
        except SyntaxError:
            styles_list = (
                styles_str.replace("[", "").replace("]", "").split(",")
            )
            return f"{[s.strip() for s in styles_list]}"

    df["style_attributes"] = df["style_attributes"].apply(
        parse_style_attributes
    )

    def parse_price(price_str):
        clean_price = price_str.replace("$", "").replace("USD", "")
        try:
            return float(clean_price)
        except ValueError:
            return -1.0

    df["price"] = df["price"].apply(parse_price)
    df = df[df["price"] > 0]
    mean_prices = df.groupby(["product_name", "brand_name"])["price"].mean()
    df["price"] = df.set_index(["product_name", "brand_name"]).index.map(
        mean_prices
    )
    df = df.drop_duplicates()
    brand_mapping = {
        "HankyPanky": "Hanky Panky",
        "HANKY PANKY": "Hanky Panky",
        "Calvin-Klein": "Calvin Klein",
        "CALVIN KLEIN": "Calvin Klein",
        "Calvin Klein Modern Cotton": "Calvin Klein",
        "b.tempt'd by Wacoal": "Wacoal",
        "B.TEMPT'D BY WACOAL": "Wacoal",
        "WACOAL": "Wacoal",
        "Victoria's Secret Pink": "Victoria's Secret",
        "Victorias-Secret": "Victoria's Secret",
    }
    df["brand_name"] = df["brand_name"].replace(brand_mapping)
    brand_counts = df["brand_name"].value_counts()
    brands_with_100_plus = brand_counts[brand_counts >= 100].index
    final_df = df[df["brand_name"].isin(brands_with_100_plus)]
    with open(preprocessed_dataset.path, "w") as f:
        final_df.to_csv(f)


@component(
    base_image="python:3.11.7",
    packages_to_install=["google-cloud-aiplatform==1.74.0"],
)
def train_model(preprocessed_dataset: Input[Dataset], model: Output[Model]):
    from google.cloud import aiplatform

    PROJECT_ID = "challenge-443804"
    REGION = "us-central1"

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
    )

    train_job = aiplatform.CustomJob(
        project=PROJECT_ID,
        location=REGION,
        display_name="model-training",
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": "n1-standard-4"},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "us-central1-docker.pkg.dev/challenge-443804/linguerie-repository/training",
                    "command": [
                        "python",
                        "train.py",
                        "--output_path",
                        model.uri,
                        "--dataset_path",
                        preprocessed_dataset.path,
                    ],
                },
            }
        ],
        staging_bucket="gs://linguerie-training",
    )
    train_job.run()


@component(
    base_image="python:3.11.7",
    packages_to_install=["google-cloud-aiplatform==1.74.0"],
)
def register_model(model: Input[Model]) -> str:
    from google.cloud import aiplatform

    PROJECT_ID = "challenge-443804"
    REGION = "us-central1"
    aiplatform.init(project=PROJECT_ID, location=REGION)

    uploaded_model = aiplatform.Model.upload(
        display_name="linguerie-regression-model",
        artifact_uri=model.uri,
        serving_container_image_uri=f"us-central1-docker.pkg.dev/{PROJECT_ID}/linguerie-repository/serving",
    )
    return uploaded_model.resource_name


@component(
    base_image="python:3.11.7",
    packages_to_install=["google-cloud-aiplatform==1.74.0"],
)
def create_and_deploy_model_to_endpoint(
    model_resource_name: str,
) -> str:
    PROJECT_ID = "challenge-443804"
    REGION = "us-central1"
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION)

    endpoint = aiplatform.Endpoint.create(display_name="supreme endpoint")

    model = aiplatform.Model(model_resource_name)
    deployed_model = model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1
    )

    return deployed_model.resource_name


@dsl.pipeline(
    name="linguerie-model-pipeline",
    description="An example pipeline that processes data and trains a model on Vertex AI.",
)
def my_pipeline():
    ingest_data_op = ingest_data()
    preprocess_op = preprocess_data(
        linguerie_dataset=ingest_data_op.outputs["linguerie_dataset"]
    )
    train_op = train_model(
        preprocessed_dataset=preprocess_op.outputs["preprocessed_dataset"]
    )
    register_op = register_model(model=train_op.outputs["model"])
    create_and_deploy_model_op = create_and_deploy_model_to_endpoint(
        model_resource_name=register_op.output
    )


compiler.Compiler().compile(
    pipeline_func=my_pipeline, package_path="my_pipeline_job.json"
)

pipeline_root_path = "gs://linguerie-training/pipeline_root"


aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
)

job = aiplatform.PipelineJob(
    display_name="This pipeline trains and deploys a regression model for linguerie price prediction",
    enable_caching=False,
    template_path="my_pipeline_job.json",
    pipeline_root=pipeline_root_path,
)

job.submit()
