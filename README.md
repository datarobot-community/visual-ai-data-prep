# Visual AI Data Preparation

Some tools to simplify and automate preparing datasets for Visual AI.

Main functions are:
- Convert folder of images to base64 csv (useful for predictions)
- Downscale images to the size used by DataRobot (useful for large datasets)
- Download images by URLs

You can find more details and examples in [this community post](https://community.datarobot.com/t5/resources/getting-predictions-for-visual-ai-projects-via-api-calls/ta-p/10864)

## Usage

Basic usage is:

    python visualai_data_prep.py <input.csv> <output.csv> <image_col>

For more details, refer to `visualai_data_prep.py`.


## Setup/Installation

    pip install Pillow
    pip install boto3

## Development and Contributing

If you'd like to report an issue or bug, suggest improvements, or contribute code to this project, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).


# Code of Conduct

This project has adopted the Contributor Covenant for its Code of Conduct.
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) to read it in full.

# License

Licensed under the Apache License 2.0.
See [LICENSE](LICENSE) to read it in full.


