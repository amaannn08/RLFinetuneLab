"""
Pydantic v2 configuration schema for RLFinetuneLab.
Provides strict validation, sensible defaults, and stage-specific configurations.
"""

from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class QuantizationConfig(BaseModel):
    """Configuration for 4-bit and 8-bit QLoRA precision quantization."""
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True

    @model_validator(mode="after")
    def validate_quantization_mutual_exclusion(self) -> "QuantizationConfig":
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot set both load_in_4bit and load_in_8bit to True.")
        return self


class LoraConfigSchema(BaseModel):
    """LoRA parameter-efficient fine-tuning configuration."""
    r: int = Field(default=16, ge=1, description="LoRA attention dimension rank.")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA scaling parameter.")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout rate.")
    target_modules: Union[List[str], str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="List of module names or 'all-linear' to apply LoRA."
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


class ModelConfig(BaseModel):
    """Base model and tokenizer loading specification."""
    name_or_path: str = Field(..., description="HF repo ID or local checkpoint path.")
    tokenizer_name_or_path: Optional[str] = Field(default=None, description="Tokenizer path if different.")
    torch_dtype: Literal["bfloat16", "float16", "float32", "auto"] = "bfloat16"
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "sdpa"
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = True
    device_map: Optional[str] = "auto"


class DatasetConfig(BaseModel):
    """Dataset ingestion and formatting specification."""
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    dataset_split: str = "train"
    prompt_field: str = "prompt"
    response_field: str = "response"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    system_prompt: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    format_type: Literal["chatml", "alpaca", "sharegpt", "custom"] = "chatml"


class SFTTrainingConfig(BaseModel):
    """Supervised Fine-Tuning stage configuration."""
    output_dir: str = "outputs/sft"
    num_train_epochs: float = Field(default=3.0, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    per_device_eval_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_seq_length: int = Field(default=2048, ge=64)
    gradient_checkpointing: bool = True
    packing: bool = False
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: Optional[int] = Field(default=100, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    report_to: List[str] = Field(default_factory=lambda: ["tensorboard"])
    resume_from_checkpoint: Optional[str] = None


class DPOTrainingConfig(BaseModel):
    """Direct Preference Optimization stage configuration."""
    output_dir: str = "outputs/dpo"
    num_train_epochs: float = Field(default=2.0, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    per_device_eval_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=5e-6, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    beta: float = Field(default=0.1, gt=0, description="DPO implicit reward temperature.")
    loss_type: Literal["sigmoid", "ipo", "kto_pair", "bco_pair"] = "sigmoid"
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5)
    max_prompt_length: int = Field(default=1024, ge=32)
    max_target_length: int = Field(default=1024, ge=32)
    max_seq_length: int = Field(default=2048, ge=64)
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=50, ge=1)
    eval_steps: Optional[int] = Field(default=50, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    report_to: List[str] = Field(default_factory=lambda: ["tensorboard"])
    reference_free: bool = False
    resume_from_checkpoint: Optional[str] = None


class GRPOTrainingConfig(BaseModel):
    """Group Relative Policy Optimization (DeepSeek-style) stage configuration."""
    output_dir: str = "outputs/grpo"
    num_train_epochs: float = Field(default=1.0, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-6, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    num_generations: int = Field(default=4, ge=2, description="Number of completions per prompt (group size G).")
    max_prompt_length: int = Field(default=512, ge=32)
    max_completion_length: int = Field(default=512, ge=32)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    beta: float = Field(default=0.04, ge=0.0, description="KL divergence coefficient vs reference model.")
    epsilon: float = Field(default=0.2, gt=0.0, le=1.0, description="PPO clipping parameter.")
    reward_funcs: List[str] = Field(
        default_factory=lambda: ["accuracy", "format"],
        description="Names of registered reward functions."
    )
    reward_weights: Optional[List[float]] = None
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=50, ge=1)
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    report_to: List[str] = Field(default_factory=lambda: ["tensorboard"])
    resume_from_checkpoint: Optional[str] = None


class EvalConfig(BaseModel):
    """Evaluation harness configuration."""
    metrics: List[str] = Field(default_factory=lambda: ["perplexity", "generation"])
    eval_batch_size: int = Field(default=4, ge=1)
    max_new_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    judge_model: Optional[str] = None
    num_samples: Optional[int] = 100


class ServingConfig(BaseModel):
    """Serving and inference engine configuration (vLLM / HuggingFace)."""
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=1.0)
    max_model_len: int = Field(default=4096, ge=128)
    quantization: Optional[str] = None
    dtype: str = "auto"


class PipelineConfig(BaseModel):
    """Master pipeline configuration unifying all stages."""
    stage: Literal["sft", "dpo", "grpo", "eval", "export", "serve", "smoke"] = "sft"
    project_name: str = "RLFinetuneLab"
    run_name: Optional[str] = None
    model: ModelConfig
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfigSchema = Field(default_factory=LoraConfigSchema)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    sft: Optional[SFTTrainingConfig] = None
    dpo: Optional[DPOTrainingConfig] = None
    grpo: Optional[GRPOTrainingConfig] = None
    eval: Optional[EvalConfig] = None
    serving: Optional[ServingConfig] = None

    @model_validator(mode="after")
    def validate_stage_config(self) -> "PipelineConfig":
        if self.stage == "sft" and self.sft is None:
            self.sft = SFTTrainingConfig()
        elif self.stage == "dpo" and self.dpo is None:
            self.dpo = DPOTrainingConfig()
        elif self.stage == "grpo" and self.grpo is None:
            self.grpo = GRPOTrainingConfig()
        elif self.stage == "eval" and self.eval is None:
            self.eval = EvalConfig()
        elif self.stage == "serve" and self.serving is None:
            self.serving = ServingConfig()
        return self
