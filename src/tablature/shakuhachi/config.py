from pydantic import BaseModel, Field

class ShakuhachiConfig(BaseModel):
    """Configuration for Shakuhachi OMR."""

    model_path: str = Field(
        'trained_models/shakuhachi_model.sav',
        description='Path to the trained model file')
    dataset_path: str = Field(
        'train_data/shakuhachi',
        description='Directory containing training data')

