<div align="center">

<h1>BEVBert: Multimodal Map Pre-training for <br /> Language-guided Navigation</h1>

<div>
    <a href='https://marsaki.github.io/' target='_blank'>Dong An</a>;
    <a href='https://sites.google.com/site/yuankiqi/home' target='_blank'>Yuankai Qi</a>;
    <a>Yangguang Li</a>;
    <a href='https://yanrockhuang.github.io/' target='_blank'>Yan Huang</a>;
    <a href='http://scholar.google.com/citations?user=8kzzUboAAAAJ&hl=zh-CN' target='_blank'>Liang Wang</a>;
    <a href='https://scholar.google.com/citations?user=W-FGd_UAAAAJ&hl=en' target='_blank'>Tieniu Tan</a>;
    <a href='https://amandajshao.github.io/' target='_blank'>Jing Shao</a>;
</div>

<h3><strong>Accepted to <a href='https://iccv2023.thecvf.com/' target='_blank'>ICCV 2023</a></strong></h3>

<h3 align="center">
  <a href="xxx" target='_blank'>Paper</a>
</h3>
</div>

## Abstract

Large-scale pre-training has shown promising results on the vision-and-language navigation (VLN) task. However, most existing pre-training methods employ discrete panoramas to learn visual-textual associations. This requires the model to implicitly correlate incomplete, duplicate observations within the panoramas, which may impair an agent’s spatial understanding. Thus, we propose a new map-based pre-training paradigm that is spatial-aware for use in VLN. Concretely, we build a local metric map to explicitly aggregate incomplete observations and remove duplicates, while modeling navigation dependency in a global topological map. This hybrid design can balance the demand of VLN for both short-term reasoning and long-term planning. Then, based on the hybrid map, we devise a pre-training framework to learn a multimodal map representation, which enhances spatial-aware cross-modal reasoning thereby facilitating the language-guided navigation goal. Extensive experiments demonstrate the effectiveness of the map-based pre-training route for VLN, and the proposed method achieves state-ofthe-art on four VLN benchmarks (R2R, RxR, REVERIE, R2R-CE).

## Method

![](assets/method.png)

Code coming soon!
