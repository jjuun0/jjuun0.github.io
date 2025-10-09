---
title: Wan-Alpha 논문 리뷰
date: 2025-10-07
categories: [논문]
tags: [Wan, RGBA, Wan-Alpha, video generation]
# use_math: true
math: true
---

# Wan-Alpha: High-Quality Text-to-Video Generation with Alpha Channel
> [arxiv](https://www.arxiv.org/abs/2509.24979)  
> Dong, H., Wang, W., Li, C., & Lin, D  
> Tianjin University, China
> 25.09

![img](assets/img/post/2025-10-07/Fig-1.png)

## Abstract
- **RGBA 비디오 생성**은 일반적인 RGB 색상 채널에 투명도를 나타내는 alpha 채널을 추가한 형태임.
- Wan-Alpha 는 RGB 와 alpha 채널을 **jointly 하게 학습**하여 비디오를 생성하는 새로운 프레임워크를 제안함.
- **VAE 가 alpha 채널을 인코딩해서 RGB latent space 에 임베딩**하도록 설계함.
- 고품질의 다양한 RGBA 비디오 데이터셋을 직접 구축함.

## 1. Introduction
- T2V(텍스트-비디오 생성)는 다양한 산업(영화, TV 제작, 게임, 가상 현실, 소셜미디어, 광고)에서 상당한 잠재력을 입증함.
- RGBA 비디오는 다양한 어플리케이션(비디오 편집, 게임 발전, 소셜 미디어 콘텐츠)에서 유용하지만, 정작 RGBA 비디오 생성은 많은 주목을 받지 못했음.
- 가장 큰 이유는 **RGBA 데이터가 RGB 데이터보다 훨씬 희귀하고 수집하기 어려워** 대규모 RGBA 비디오 생성 모델을 학습하는 것이 챌린지가 되었음.
    - 연구자들은 pretrained RGB 생성 모델을 활용하는 방법을 시도함.
    - **LayerDiffuse**: 첫 번째 RGBA 이미지 생성 모델로, VAE를 사용해 투명도 정보를 RGB latent space 에 임베딩했음.(LoRA 학습 방식 적용)
    - **AlphaVAE**: RGBA VAE를 개선하면서 학습 데이터 수를 100만 장에서 8천 장으로 줄였음.
    - **Alfie**: training-free 방법 제안함.(attention maps 에서 alpha 값을 직접 추론했음.)
        - 하지만, 작은 객체는 잘 생성하지 못한 한계가 있었음.    
- 그 뒤에도 multi-layer 생성 방식이나 inpainting 방식으로 투명도를 생성하는 연구가 지속됨.  
    - **TransPixeler**: 최신 SOTA 모델로 alpha tokens 개념을 도입하고 백본 네트워크를 복제했으며 cross-RGBA attention 방식을 적용했음.
        - 하지만, inference cost가 두 배로 증가했고, 결과도 만족스럽지 못했음.
- Wan-Alpha
    1. 고품질 데이터셋 구축
        - Wan 모델을 베이스 모델로 선정하고, visual quality 향상에 초점을 맞췄음.
        - 고해상도, 부드러운 움직임, 다양한 콘텐츠를 보장하기 위해 여러 출처에서 데이터를 수집했음.
        - **semantic consistency** 를 유지하기 위해 비디오 콘텐츠랑 프롬프트가 정확히 일치하는지 확인함.
    2. 간단하고 효율적인 학습 및 추론 과정
        - **LayerDiffuse** 전략을 채택했음.(VAE 를 학습하여 alpha 정보를 RGB latent space 에 임베딩함.)
        - **Feature merge block** 을 causal 3D 모듈과 함께 사용해 RGB 와 alpha latent feature를 효과적으로 결합함.
        - RGB/alpha decoder는 **LoRA** 방식을, DiT(diffusion transformer)는 **DoRA** 모듈을 적용하여 학습함.
        - inference 에서는 모델을 두 개의 LoRA 모듈(VAE decoders) 및 DoRA 모듈(DiT)을 로드함.
        - TransPixeler 와 비교해 추가 연산 비용 없이, 4-step 에서도 CFG 없이 inference 가능함.

## 2. Method
- [2.1 VAE 설계](#21-vae): alpha 채널을 RGB latent space 에 어떻게 인코딩하는지 상세히 다룸.
- [2.2 Text-to-Video 생성](#22-text-to-video-generation): pretrained VAE를 활용해 DiT를 어떻게 학습했는지 설명하며, 특히 데이터 수집과 inference acceleration 을 소개함.

### 2.1 VAE
- Wan-Alpha 는 LayerDiffuse 의 접근 방식을 따라, **RGB-alpha VAE 가 alpha 채널을 RGB VAE 와 동일한 latent space 로 인코딩**하도록 설계했음.
    - LayerDiffuse 에서는 별도의 encoder 와 decoder 를 위한 UNet 구조를 사용하는데, 이는 비디오 도메인에서 더 학습하기 어려운 문제가 있음.
    - 이를 해결하기 위해 **pretrained VAE를 복제**하여 사용하는 전략을 채택함.
    - encoder 에는 RGB-alpha feature merging block를 도입했고, RGB decoder 및 alpha decoder를 효율적으로 조정하기 위해 LoRA를 사용했음.
    - 또한 VAE 학습을 더욱 효과적으로 수행하기 위해 일련의 렌더링 기법 및 loss function 을 도입했음.

- **Architecture**
    - 모델 설계전에 두 개의 함수를 정의함
        - soft render $\mathbb{R}^s$, hard render $\mathbb{R}^h$ 


    $$  
    \begin{aligned}
    & \mathbb{R}^s=V_{r g b} \cdot \alpha+c \cdot(1-\alpha), \\
    & \mathbb{R}^h=V_{r g b} \cdot \alpha+c \cdot(1-\alpha), \quad \alpha= \begin{cases}1 & \text { if } \alpha>0, \\
    0 & \text { otherwise. }\end{cases}
    \end{aligned}
    $$  

    - $c$: 사전에 정의된 colors set $C$ 중 랜덤 색상임.
        - $C = \text{\{black, blue, green, cyan, red, magenta, yellow, white\}}$
    - **Soft rendering**: alpha 비디오 값이 $[0,1]$ 범위에 속하는 과정을 의미하며, 이는 일반적으로 **RGBA 비디오의 배경을 자연스럽게 합성**하는데 사용됨.
    - **Hard rendering**: 반대로, **0이 아닌 모든 alpha 값을 1로 설정**하고 완전히 투명하지 않은 영역은 RGB 채널의 원본 색상을 그대로 유지함.
    
    ![img](assets/img/post/2025-10-07/Fig-2.png)

    - RGBA 비디오를 RGB 비디오와 alpha 비디오로 분리하고, **alpha 비디오 채널은 3배 복제**함.  
    - encoder가 RGB 배경색과 투명도를 혼동하는것을 방지하기 위해, color set $C$ 로 부터 랜덤하게 color $\bar{c}$ 를 선택하고, RGB 비디오 $V_{r g b}$ 를 hard-render 하여 rendered 비디오 $\bar{V}_{r g b}$ 를 생성함.  


    $$  
    \bar{V}_{r g b}=\mathbb{R}^h\left(V_{r g b}, V_\alpha, \bar{c}\right)
    $$  


> 개인 해석: 투명 영역 전처리의 중요성    
> 인코더가 RGB, alpha 를 모두 입력받는데, 둘을 명확하게 구분짓기 위해 투명 영역을 매번 다른 색상으로 채워넣습니다.  
> 원본 RGB 영상에서는 투명한 영역에는 시각적으로 보이지 않지만, 실제 데이터는 어떤 RGB값이 저장되어 있습니다.  
> 모델은 저장된 값으로 그대로 처리를 하기 때문에 투명도를 혼동하게 됩니다.  
> 따라서, Wan-Alpha 에서는 매번 랜덤한 색상으로 채워, 항상 다른 배경색을 가지게 됩니다.  
> 결과적으로 모델은 RGB 색상만으로 투명도를 예측할 수 없게 되고, 반드시 alpha 채널 정보를 활용해야 정확한 투명도를 학습할 수 있게 됩니다.   
{: .prompt-tip }

![img](assets/img/post/2025-10-07/Fig-3.png)

$$
Z=\mathcal{M}\left(\mathcal{E}\left(\bar{V}_{r g b}\right), \mathcal{E}\left(V_\alpha\right)\right)
$$  

- $\bar{V}\_{r g b}$ 와 $V_\alpha$ 를 각각 frozen Wan-VAE encoder $\mathcal{E}$ 에 입력함.  
- 이후 **3D causal convolution** 모듈을 사용해 feature merging block $\mathcal{M}$ 을 구성함.  
    - 일반적인 3D convolution 과 비교했을 때, 3D causal convolution 은 연산 비용이 더 적고 더 긴 비디오를 인코딩할 수 있음.
    - feature merging block $\mathcal{M}$은 causal residual blocks와 attention layers 로 구성됨.
    - 처음부터 학습하지 않고 pretrained 모델로 파인튜닝하기 때문에, **latent space 분산이 커지는 것을 억제할 필요가 없음**.
        - 일반적으로 VAE를 학습할 때는 KL divergence term을 추가해 latent space 분산을 제어하는데,
        - 여기서는 pretrained(이미 안정된 latent space)를 사용하기 때문에, feature merge block 는 **KL penalty 를 사용하지 않고** 바로 latent $Z$ 를 예측함.

    - 구체적인 처리과정
        1. 렌더링된 $\bar{V}\_{rgb}$ 와 ${V}_{\alpha}$ 가 각각 인코딩된 feature로 입력됨.  
        2. 두 feature를 concat한 후, causal 3D convolution 에 통과시킴.
        3. causal residual block 및 attention 연산을 거쳐 latent feature $Z$ 생성함.
    - 최종적으로 latent $Z$ 는 **frozen Wan-VAE decoders**(RGB LoRA 및 alpha LoRA 가 포함.)에 입력됨.

        $$
        \hat{V}_{r g b}=\mathcal{D}_{\mathrm{w} / \mathrm{RGB} \text { LoRA }}(Z), \quad \hat{V}_\alpha=\mathcal{D}_{\mathrm{w} / \text { Alpha LoRA }}(Z) .
        $$

        - reconstructed RGB video $\hat{V}\_{r g b}$ 와 $\hat{V}_\alpha$ 를 얻음.

- **Training Objectives**
    - 3 개의 training objectives 사용함.
    - alpha channel $\hat{V}\_\alpha$, soft-rendered video $\hat{V}^{s}\_{r g b}$, hard-rendered video $\hat{V}^{h}_{r g b}$
        
        $$
        \hat{V}_{r g b}^s=\mathbb{R}^s\left(\hat{V}_{r g b}, \hat{V}_\alpha, c_s\right), \quad \hat{V}_{r g b}^h=\mathbb{R}^h\left(\hat{V}_{r g b}, \hat{V}_\alpha, c_h\right),
        $$
        
        - soft-rendered video: alpha 채널과 RGB 채널에 대한 통합된 가이드(joint guidance)를 제공함.
        - hard-rendered video: RGB 채널에 좀 더 중점을 두며, 반투명 영역의 투명도는 무시함.
        - $c_s$ 와 $c_h$ 는 color set $C$ 의 랜덤 색상임.

    1. **reconstructed loss**  

        $$
        \mathcal{L}_{r e c}\left(\hat{V}_\alpha, V_\alpha\right)=\left\|\hat{V}_\alpha-V_\alpha\right\| .
        $$  

    2. **perceptual loss**
        - VGG 네트워크 $\Phi$

        $$
        \mathcal{L}_{p e r}\left(\hat{V}_\alpha, V_\alpha\right)=\left\|\Phi\left(\hat{V}_\alpha\right)-\Phi\left(V_\alpha\right)\right\|_2
        $$

    3. **edge loss**  
        - Sobel operator $S(\cdot)$: edge gradients 추출.

        $$
        \mathcal{L}_{\text {edge }}\left(\hat{V}_\alpha, V_\alpha\right)=\left\|S\left(\hat{V}_\alpha\right)-S\left(V_\alpha\right)\right\| .
        $$
    
    - 이렇게 alpha video loss 가 구성됨.

    $$
    \mathcal{L}_\alpha=\mathcal{L}_{r e c}\left(\hat{V}_\alpha, V_\alpha\right)+\mathcal{L}_{p e r}\left(\hat{V}_\alpha, V_\alpha\right)+\mathcal{L}_{e d g e}\left(\hat{V}_\alpha, V_\alpha\right)
    $$

    - 또한 soft-rendered, hard-rendered video 에 대해서도 loss 를 측정함.

    $$
    \begin{aligned}
    \mathcal{L}_{r g b}^s & =\mathcal{L}_{r e c}\left(\hat{V}_{r g b}^s, V_{r g b}^s\right)+\mathcal{L}_{p e r}\left(\hat{V}_{r g b}^s, V_{r g b}^s\right)+\mathcal{L}_{e d g e}\left(\hat{V}_{r g b}^s, V_{r g b}^s\right), \\
    \mathcal{L}_{r g b}^h & =\mathcal{L}_{r e c}\left(\hat{V}_{r g b}^h, V_{r g b}^h\right)+\mathcal{L}_{p e r}\left(\hat{V}_{r g b}^h, V_{r g b}^h\right)+\mathcal{L}_{e d g e}\left(\hat{V}_{r g b}^h, V_{r g b}^h\right) .
    \end{aligned}
    $$

    - total loss 를 다음과 같이 구성함.

    $$
    \mathcal{L}_{v a e}=\mathcal{L}_\alpha+\mathcal{L}_{r g b}^s+\mathcal{L}_{r g b}^h .
    $$


### 2.2 Text-to-Video Generation
- VAE를 통해 RGBA 비디오를 RGB 비디오의 latent space와 동일하게 매핑할 수 있게 되었음.
- 하지만, 여전히 **생성 비디오의 품질을 향상시켜야하는** 과제가 남아있음.

![img](assets/img/post/2025-10-07/Fig-4.png)

- **Architecture**
    - Wan 모델을 베이스 모델로 채택함.
    - transparent video $V$는 VAE encoder $\mathcal{E}$ 및 feature merge block $\mathcal{M}$를 거쳐 latent representation $Z$ 로 압축됨.
    - text condition 은 T5 text encoder 로 인코딩됨.
    - DiT 는 **DoRA** 를 적용해 학습함.(LoRA와 비교했을 때, semantic alignment 와 고품질 비디오 생성 측면에서 더 우수한 성능을 보였음.)

> 참고: LoRA vs DoRA  
> LoRA: $W = W\_0 + \Delta W = W\_0 + BA$  
> DoRA: $W = (\left \lVert W\_0 \right \rVert + \Delta m) \cdot \mathrm{normalize}(W\_0 + \Delta W\_d)$  
> LoRA 는 $W\_0$ 가 임의로 커질 수 있어서 weight scale이 예측 불가능하게 변동됩니다.  
> 이로 인해 학습이 불안정하거나 성능이 떨어질 수 있습니다.  
> 하지만, DoRA 는 $\mathrm{nomalize}()$ 로 방향을 일정하게 유지하며, $\Delta m$ 으로만 크기를 제어합니다.  
> 이렇게 하면 $\left \lVert W \right \rVert$의 변화는 오직 $\Delta m$ 를 통해서만 발생하므로, scale 이 예측 가능한 방식으로 조정됩니다.  
> 결과적으로 DoRA는 안정적인 scale 을 유지하면서 더욱 안정화된 학습을 가능하게 합니다.    
{: .prompt-tip }

- **Training Objective**
    - **Rectified Flow** 접근 방식을 따랐음.
        - 이는 데이터 분포를 노이즈 분포로 매핑하는 과정을 **직선 경로**로 구성하는 방식임.
    
    $$
    Z_t = t \epsilon + (1 - t)Z
    $$

    - target velocity

    $$
    v_t = \epsilon - Z
    $$

    - 예측된 속도 $\hat{v}_t$ 와 GT 를 mean squared error 로 최소화하는 것으로 loss 구성함.

    $$
    \mathcal{L}_{t2v} = \left \lVert \hat{v}_t - v_t \right \rVert ^2
    $$

- **Dataset**
    - 데이터 선정
        - 고품질 데이터를 확보하기 위해 선별 과정 거침.
            - **명확한 모션, 반투명 객체, 조명 효과**가 뚜렷한 데이터 선택함.
    - 캡션 생성 과정
        - Qwen2.5-VL-72B 모델 사용해 짧은 캡션과 긴 캡션을 자동으로 생성하고, 수동으로 수정했음.
        - 모든 캡션은 중국어로 작성되었지만, Wan 모델의 다국어 지원 능력 덕분에 중국어 데이터만으로 학습했어도 영어 프롬프트로 RGBA 비디오 생성 가능했음.
    - 추가 라벨링
        - 수동으로 모션 속도, 예술적 스타일, 샷 크기, 시각적 품질 문제 등 라벨을 부여함.
        - 캡션의 앞 또는 뒤에 랜덤으로 삽입했음.

- **Inference**
    - 기존의 Wan 모델을 기반으로 사용자는 오직 **두 개의 VAE decoder LoRAs 와 t2v DoRA** 만 로드하면 됨.
    - LoRA/DoRA weights 는 완전히 베이스 모델에 병합 가능하며 추가 연산 비용 없음.
    - inference pipeline 은 간단한 변화가 있었음.
        - decoder 를 복제하여 하나는 RGB 채널을, 다른 하나는 alpha 채널을 생성하도록 구성함.
        - 구조가 단순해지고 배포 및 가속화가 쉬워졌음.
    - 가속화는 LightX2V(CFG 없이, 4 steps 로 high-quality 결과를 생성할 수 있음)을 사용했음.


### 3.1 Implementation Details
- resolution $480 \times 832$
- 81 frames(16 FPS)
- 4 steps
- LoRA rank(RGB, alpha VAE decoder): 128
- DoRA rank(DiT): 32
- VAE: 60,000 iterations, batch size 2
- DiT: 1,500 iterations, batch size 8

### 3.2 Comparisons
- 비교 대상
    - TransPixeler(open): CogVideoX-5B 기반 오픈소스 버전
    - TransPixeler(close): Adobe Firefly를 통해 접근한 closed 버전
- 테스트 프롬프트는 대부분 TransPixeler 논문에서 사용한 것으로 사용함.

- 결과
    - Wan-Alpha 는 TransPixeler 대비 **모션 일관성, 시각적 품질, 알파 엣지 선명도, 반투명 효과**에서 우수한 성능을 보였음.
    - inference 속도는 TransPixeler(open) 대비 약 **15배** 빠름.


> RGBA 비디오의 alpha 채널을 R**GB latent space 에 동일하게 매핑**시키려고 한점이 핵심 아이디어였던 것 같습니다.  
> 특히 주목할 점은 데이터 전처리 전략입니다. **frozen VAE encoder**를 RGB와 alpha 입력 모두에 사용하기 위해 **soft/hard rendering**을 적용하는 부분을 잘 이해하면 다른 작업에서 응용할 수 있을 것 같습니다.  
> **학습 데이터를 테스크에 맞게 잘 가공하는 것**이 얼마나 중요한지 다시 깨닫게 되었습니다.  
> 참고로 Wan 모델도 최근 비디오 생성 연구에 베이스 모델로 자주 활용되고 있어, 관련 논문도 함께 읽어보면 도움이 될 것 같습니다.  
{: .prompt-info }