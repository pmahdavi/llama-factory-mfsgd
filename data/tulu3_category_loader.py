import datasets
import json

# Define the mapping from categories to source keys based on the analysis
_CATEGORY_SOURCES = {
    "general": [
        "ai2-adapt-dev/oasst1_converted",
        "ai2-adapt-dev/tulu_hard_coded_repeated_10",
        "ai2-adapt-dev/no_robots_converted",
        "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    ],
    "knowledge_recall": [
        "ai2-adapt-dev/flan_v2_converted",
        "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
        "ai2-adapt-dev/tulu_v3.9_table_gpt_5k",
    ],
    "math_reasoning": [
        "ai2-adapt-dev/personahub_math_v5_regen_149960",
        "allenai/tulu-3-sft-personas-math-grade",
        "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
        "ai2-adapt-dev/numinamath_tir_math_decontaminated",
        "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
    ],
    "coding": [
        "ai2-adapt-dev/personahub_code_v2_34999",
        "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
    ],
    "safety_and_compliance": [
        "ai2-adapt-dev/coconot_converted",
        "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
        "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
    ],
    "multilingual": [
        "ai2-adapt-dev/tulu_v3.9_aya_100k",
    ],
    "precise_if": [
        "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
    ],
}

class Tulu3CategoryConfig(datasets.BuilderConfig):
    """BuilderConfig for Tulu3 categories."""
    def __init__(self, category=None, **kwargs):
        """Initializes a Tulu3CategoryConfig.

        Args:
            category (str): The category to filter by.
            **kwargs: keyword arguments forwarded to super.
        """
        super(Tulu3CategoryConfig, self).__init__(**kwargs)
        self.category = category

class Tulu3CategoryDatasetBuilder(datasets.GeneratorBasedBuilder):
    """A DatasetBuilder for loading and filtering the Tulu-3 SFT mixture by category."""

    # Define BUILDER_CONFIGS based on the keys in _CATEGORY_SOURCES
    BUILDER_CONFIGS = [
        Tulu3CategoryConfig(
            name=category,
            version=datasets.Version("1.0.0"),
            description=f"Tulu-3 SFT mixture filtered for the '{category}' category.",
            category=category
        )
        for category in _CATEGORY_SOURCES.keys()
    ]

    DEFAULT_CONFIG_NAME = "general" # Optional: set a default config

    def _info(self):
        # Define the features based on the expected structure of "allenai/tulu-3-sft-mixture"
        # This needs to match the actual structure, especially the 'messages' and 'source' columns.
        return datasets.DatasetInfo(
            description=f"Tulu-3 SFT mixture dataset filtered for category: {self.config.category}",
            features=datasets.Features(
                {
                    "messages": [
                        {
                            "role": datasets.Value("string"),
                            "content": datasets.Value("string"),
                        }
                    ],
                    "id": datasets.Value("string"), # Assuming 'id' is a string
                    "source": datasets.Value("string"), # Assuming 'source' is a string
                    # Add other features if they exist and are needed
                }
            ),
            homepage="https://huggingface.co/datasets/allenai/tulu-3-sft-mixture",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # The dataset is loaded directly from Hugging Face Hub, so no download/extraction needed here.
        # The category is accessed via self.config.category
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These are dummy gen_kwargs, actual loading happens in _generate_examples
                gen_kwargs={"split": "train", "category": self.config.category},
            ),
            # Add other splits if necessary, e.g., VALIDATION, TEST
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     gen_kwargs={"split": "validation", "category": self.config.category},
            # ),
        ]

    def _generate_examples(self, split, category):
        """Yields examples."""
        print(f"Generating examples for category '{category}', split '{split}'...")

        if category not in _CATEGORY_SOURCES:
            raise ValueError(
                f"Invalid category '{category}'. Available categories: {list(_CATEGORY_SOURCES.keys())}"
            )
        target_sources = set(_CATEGORY_SOURCES[category])

        try:
            # Load the full dataset stream
            # Using streaming=True here for potentially large datasets
            full_dataset = datasets.load_dataset(
                "allenai/tulu-3-sft-mixture", 
                split=split, 
                streaming=True, 
                trust_remote_code=True
            )
            print(f"Successfully started streaming 'allenai/tulu-3-sft-mixture' for split '{split}'.")
        except Exception as e:
            print(f"Error loading dataset 'allenai/tulu-3-sft-mixture': {e}")
            raise

        idx = 0
        for example in full_dataset:
            if example.get("source") in target_sources:
                # Ensure the example matches the features defined in _info()
                # The 'messages' field is expected to be a list of dicts
                # If it's a string, it might need conversion (e.g., json.loads)
                # For now, assuming it's already in the correct format.
                yield idx, {
                    "messages": example.get("messages"),
                    "id": example.get("id"),
                    "source": example.get("source"),
                }
                idx += 1
        print(f"Finished generating examples for category '{category}', split '{split}'. Found {idx} examples.")
        if idx == 0:
            print(f"Warning: No examples yielded for category '{category}', split '{split}'. "
                  "Check category mapping and source data.")

# To make this script discoverable by `datasets` library, the main DatasetBuilder class
# should typically be named following conventions or be the only DatasetBuilder in the file.
# In this case, Tulu3CategoryDatasetBuilder is the class. 