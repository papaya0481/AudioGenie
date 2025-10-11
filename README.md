# 🎶 AudioGenie: A Training-Free Multi-Agent Framework for Diverse Multimodality-to-Multiaudio Generation

**This work has been accepted to ACM MM 2025! 🚀🚀🚀**

[![arXiv](https://img.shields.io/badge/arXiv-2505.22053-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2505.22053)   [![githubio](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://audiogenie.github.io/)

**This is the official repository for "[AudioGenie: A Training-Free Multi-Agent Framework for Diverse Multimodality-to-Multiaudio Generation](https://arxiv.org/pdf/2505.22053)".**


## ✨ Abstract

Multimodality-to-Multiaudio (MM2MA) generation faces significant challenges in synthesizing diverse and contextually aligned audio types (e.g., sound effects, speech, music, and songs) from multimodal inputs (e.g., video, text, images), owing to the scarcity of high-quality paired datasets and the lack of robust multi-task learning frameworks. Recently, multi-agent system shows great potential in tackling the above issues. However, directly applying it to MM2MA task presents three critical challenges: (1) inadequate fine-grained understanding of multimodal inputs (especially for video), (2) the inability of single models to handle diverse audio events, and (3) the absence of self-correction mechanisms for reliable outputs. 
To this end, we propose AudioGenie, a novel training-free multi-agent system featuring a dual-layer architecture with a generation team and a supervisor team. For the generation team, a fine-grained task decomposition and an adaptive Mixture-of-Experts (MoE) collaborative entity are designed for detailed comprehensive multimodal understanding and dynamic model selection, and a trial-and-error iterative refinement module is designed for self-correction. The supervisor team ensures temporal-spatial consistency and verifies outputs through feedback loops. Moreover, we build MA-Bench, the first benchmark for MM2MA tasks, comprising 198 annotated videos with multi-type audios. 
Experiments demonstrate that our AudioGenie achieves state-of-the-art (SOTA) or comparable performance across 9 metrics in 8 tasks. User study further validates the effectiveness of our method in terms of quality, accuracy, alignment, and aesthetic.



## ✨ Method



<p align="center"><strong>Overview of the AudioGenie Framework.</strong></p>

## 🔮 MA-Bench
The dataset will be been released on Hugging Face.

