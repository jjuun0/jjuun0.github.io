---
title: Diffusers Flux 코드 리뷰
date: 2025-10-06
categories: [코드]
tags: [Flux]
---

> 버전: 0.35.1 (2025.10.06 기준)  
> [github](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py), [huggingface docs](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/flux#flux) 참고

앞선 flux-kontext 의 베이스 모델인 **flux** 코드를 살펴보고 **stable diffusion v1.5** 와 어떻게 달라졌는지 비교해보겠습니다.
디테일한 점보다는, 큰 흐름에서 변경된 점만 빠르게 파악해보겠습니다.

## Inference
- 우선, inference 코드부터 살펴보겠습니다.

{% highlight python linenos %}
import torch  
from diffusers import FluxPipeline   

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", \
                                    torch_dtype=torch.bfloat16)  
pipe.to("cuda")  
prompt = "A cat holding a sign that says hello world"  
# Depending on the variant being used, the pipeline call will slightly vary.  
# Refer to the pipeline documentation for more details.  
image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]  
image.save("flux.png")  
{% endhighlight %}

- pipeline 만 달라서 **FluxPipeline** 내부를 살펴보겠습니다.  

## 입력 파라미터  
- pipe() 가 실행되면, __call__ 가 호출되는데, 이 함수에 필요한 입력을 살펴보겠습니다.

```python
def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
```


- prompt 는 prompt_2 가 추가되었습니다.  
- guidance_scale 이외의 true_cfg_scale 이 추가로 입력됩니다. 

## true_cfg_scale 처리  
- true_cfg_scale은 기본 guidance_scale과 별개로 negative prompt 관련 CFG 보정을 정밀하게 제어하기 위해 도입되었습니다.
{% highlight python linenos %}
do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
(
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
) = self.encode_prompt(
    prompt=prompt,
    prompt_2=prompt_2,
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    device=device,
    num_images_per_prompt=num_images_per_prompt,
    max_sequence_length=max_sequence_length,
    lora_scale=lora_scale,
)
if do_true_cfg:
    (
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        negative_text_ids,
    ) = self.encode_prompt(
        prompt=negative_prompt,
        prompt_2=negative_prompt_2,
        prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
{% endhighlight %}
- true_cfg_scale 값이 1보다 크고 negative prompt 를 입력받았다면, negative_prompt_embeds 도 임베딩이 수행됩니다. 
    - 추후 denoising loop 에서 작동되는 것을 확인해보겠습니다.
- 여기서, prompt_2 도 같이 임베딩이 되는데요. encode_prompt() 함수를 잠깐 살펴보겠습니다.


## encode_prompt
- text prompt 를 text encoder 를 통해 embedding 하는 과정입니다.
{% highlight python linenos %}
def encode_prompt():
    ...
    # We only use the pooled prompt output from the CLIPTextModel
    pooled_prompt_embeds = self._get_clip_prompt_embeds(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
    )

    prompt_embeds = self._get_t5_prompt_embeds(
        prompt=prompt_2,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    ...
{% endhighlight %}
- stable diffusion 과는 다르게, flux 에서는 text encoder 를 두 개, **CLIP** 과 **T5** 모델을 사용합니다.
    - CLIP: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
        - text 입력 길이에 **77** token 까지 제한이 있었습니다.
        - **image 와 text 간의 상관관계**를 학습해 시각적인 개념과 키워드 중심의 표현을 처리하는 역할을 합니다.
    - T5: [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl)
        - **긴 문장, 복합적인 설명, 맥락적 의미 해석** 등을 처리하는 데 장점이 있습니다.
        - **512** token 까지 처리할 수 있습니다.
- 이렇게, 두 text encoder 의 각 장점을 모두 활용하기 위해 최근에는 모두 사용하는 방향으로 연구가 진행되고 있습니다. 
    - CLIP 은 텍스트의 핵심 정보만 추출하기 위해, embedding 전체를 대표할 수 있는 **pooled_output** 만 사용합니다.


{% highlight python linenos %}

# 6. Denoising loop
with self.transformer.cache_context("cond"):
    noise_pred = self.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs=self.joint_attention_kwargs,
        return_dict=False,
    )[0]

if do_true_cfg:
    if negative_image_embeds is not None:
        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

    with self.transformer.cache_context("uncond"):
        neg_noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=negative_pooled_prompt_embeds,
            encoder_hidden_states=negative_prompt_embeds,
            txt_ids=negative_text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
{% endhighlight %}

- prompt_embeds 로 denoising 해서 noise_pred 를 예측하고, 
    - do_true_cfg 가 수행될 경우, negative_prompt_embeds 로 denoising 해서 neg_noise_pred 를 예측합니다.
        - 그 후, noise_pred 를 `neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)` 로 업데이트합니다.(31번째 줄)
    - 여기서, stable diffusion 과 다른점이 있습니다.
        - stable diffusion 에서는 **prompt_embeds 와 negative_prompt_embeds 를 concat 해서 denoising 을 수행**합니다.
            - 내부 **self-attention, cross-attention 연산이 수행되어 두 가지 문맥을 동시에 참고**하면서 denoising 수행됩니다. 
        - 하지만, flux 에서는 두 embeds 를 **별도로 denoising 수행**합니다. 
            - **flux 는 flow matching 기반의 transformer 모델로 구성되어 있어, guidance 신호의 안정성을 높이기 위해** 설계된 것으로 해석됩니다.  

- 그 후, 동일하게 denoising 를 step 에 맞게 수행하여, noise 를 제거하고 decoding 을 거쳐 이미지를 생성합니다.

- 참고
    - stable diffusion 모델과 생성 결과를 비교한 블로그 [링크](https://getimg.ai/blog/flux-1-vs-stable-diffusion-ai-text-to-image-models-comparison) 참고해보시기 바랍니다.   

