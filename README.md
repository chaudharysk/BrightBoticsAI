# Deep Learning to LLMs Study Plan

Welcome to the **Deep Learning to LLMs Study Plan** repository! This curriculum is designed to guide you through an intermediate-level journey, culminating in advanced understanding and practical skills in large language models (LLMs). By the end of this program, you will be able to build, fine-tune, optimize, and deploy models like GPT, BERT, and T5 for real-world applications.

---

## 📚 **Overview**
- **Duration:** 15 December 2024 – 21 January 2025
- **Weekly Commitment:** ~20 hours (Monday–Friday, excluding Saturdays)
- **Focus Areas:**
  - Deep Learning Foundations
  - Transformer Architectures & Attention Mechanisms
  - Embeddings & Pre-Trained Models
  - Fine-Tuning & Parameter-Efficient Techniques
  - Advanced Optimization & Deployment
  - Ethical & Responsible AI

This plan integrates theory, hands-on coding exercises, and project-based learning to ensure a practical and in-depth understanding.

---

## 🔐 **Key Highlights**

1. **Hands-On Learning:**
   - Code-driven exercises with PyTorch, TensorFlow, and Hugging Face Transformers.
   - Practical examples: fine-tuning language models, building embeddings, optimizing inference.

2. **Curated Resources:**
   - Week-by-week recommended readings, tutorials, and academic papers.
   - Balanced focus on foundational theory, application, and cutting-edge research.

3. **Capstone Project:**
   - Build a full LLM-powered application (e.g., a domain-specific QA system).
   - Incorporate retrieval-augmented generation (RAG) using embeddings and vector stores.
   - Deploy the model via a web API and evaluate for performance and fairness.

4. **Ethical & Responsible AI:**
   - Considerations for bias, fairness, and transparency.
   - Strategies for alignment and RLHF to ensure models serve beneficial purposes.

---

## 🛠️ **Skills You Will Gain**
- Neural network fundamentals and optimization techniques.
- Implementing attention, building Transformers, and understanding the core architecture of LLMs.
- Working with embeddings (Word2Vec, GloVe, BERT, GPT-based embeddings) and applying them in downstream tasks.
- Fine-tuning pre-trained models efficiently (LoRA, adapters) and optimizing them (mixed precision, ONNX).
- Deploying LLMs in real-world environments with vector databases and retrieval-augmented generation.
- Engaging with responsible AI principles and performing bias mitigation strategies.

---

## 📅 **Weekly Breakdown**

### Week 1: Deep Learning Foundations
**Topics:**
- Neural Networks & Backpropagation  
- Optimization Techniques (SGD, Adam, Regularization)  
- Overfitting, Underfitting, and Generalization  
- RNNs & LSTMs for Language Modeling

**Learning Objectives:**
- Understand the mathematical foundations of neural networks.
- Train and evaluate simple models on text data.
- Analyze training curves and apply basic optimization methods.

**Hands-On Deliverables:**
- Implement a simple feedforward neural network and train it on a classification task.
- Build a basic LSTM language model and observe its ability to generate text.

**Recommended Resources:**
- *Deep Learning* by Goodfellow, Bengio, and Courville (Chapters on Basics & Optimization)
- Stanford CS231n Lectures (Neural Networks Basics)
- PyTorch or TensorFlow Official Tutorials (Basic Models)

---

### Week 2: Transformers & Self-Attention
**Topics:**
- Attention Mechanisms: Scaled dot-product, Multi-Head Attention
- Transformer Architecture: Encoders, Decoders, Positional Encoding
- Tokenization Techniques: BPE, SentencePiece, WordPiece

**Learning Objectives:**
- Understand the role of attention and why it replaces recurrence.
- Implement a basic Transformer block.
- Explore tokenization strategies and their impact on downstream performance.

**Hands-On Deliverables:**
- Implement attention from scratch in `attention_demo.py`.
- Work through `annotated_transformer.ipynb` to understand Transformers step-by-step.
- Experiment with `tokenizer_experiments.py` to preprocess a custom dataset.

**Recommended Resources:**
- "Attention Is All You Need" (Vaswani et al.)
- Hugging Face Tokenizers Documentation
- The Illustrated Transformer (Online Blog)

---

### Week 3: Embeddings & Pre-Trained Models
**Topics:**
- Traditional Embeddings: Word2Vec, GloVe
- Contextual Embeddings: BERT, GPT
- Fine-Tuning Pre-Trained Models for Classification
- Prompt Engineering for Autoregressive Models

**Learning Objectives:**
- Understand embedding spaces and how word/contextual embeddings represent language semantics.
- Fine-tune BERT for downstream NLP tasks.
- Experiment with prompting GPT-based models for zero-shot or few-shot learning.

**Hands-On Deliverables:**
- Use `embeddings_demo.ipynb` to compare and contrast Word2Vec, GloVe, and BERT embeddings.
- Fine-tune BERT on a sentiment classification dataset using `bert_fine_tuning.py`.
- Implement simple prompt engineering experiments in `prompt_engineering.ipynb`.

**Recommended Resources:**
- "Distributed Representations of Words and Phrases" (Mikolov et al.)
- BERT Paper (Devlin et al.)
- Hugging Face Transformers Tutorials on Fine-Tuning

---

### Week 4: Advanced Fine-Tuning & Optimization
**Topics:**
- Parameter-Efficient Tuning: LoRA, Adapters, Prefix-Tuning
- Mixed-Precision Training and Gradient Checkpointing
- Model Compression: Pruning, Quantization, Distillation

**Learning Objectives:**
- Reduce resource requirements by applying parameter-efficient tuning methods.
- Speed up training and inference using mixed-precision and optimization tricks.
- Explore compression techniques to deploy LLMs on edge devices or constrained environments.

**Hands-On Deliverables:**
- Implement LoRA in `parameter_efficient_tuning.py` for GPT-2 fine-tuning.
- Optimize training using `mixed_precision_training.py`.
- Use `model_compression.ipynb` to prune, quantize, or distill a model and measure performance trade-offs.

**Recommended Resources:**
- LoRA Paper (Hu et al.)
- NVIDIA Mixed Precision Training Guide
- Model Distillation Techniques (Hinton et al.)

---

### Week 5: RLHF, Ethical AI & Deployment
**Topics:**
- Reinforcement Learning from Human Feedback (RLHF)
- Responsible AI: Bias detection and mitigation, Fairness, Transparency
- Deployment: ONNX optimization, FastAPI for serving, Vector Databases (FAISS) for RAG

**Learning Objectives:**
- Understand alignment strategies and RLHF to better align model behavior with human values.
- Identify bias in LLM outputs and propose mitigation strategies.
- Deploy optimized models in real-world scenarios with low latency and retrieval capabilities.

**Hands-On Deliverables:**
- Simulate RLHF workflows with `rl_hf_demo.py`.
- Optimize inference with `inference_optimization.py`.
- Explore retrieval-augmented generation workflows in `vector_store_demo.ipynb`.

**Recommended Resources:**
- "Learning to Summarize from Human Feedback" (Stiennon et al.)
- Microsoft Responsible AI Principles
- ONNX Runtime & FastAPI Documentation

---

### Capstone Project (Final Week)
**Objective:**
- Develop a fully functional LLM-powered application, such as a domain-specific question-answering system or a contextual chatbot.

**Steps:**
- Fine-Tune a chosen LLM on domain-specific data using `fine_tuning_pipeline.py`.
- Integrate retrieval-augmented generation techniques via `retrieval_augmented_generation.py`.
- Deploy the final model using `deployment_script.py` and evaluate with `evaluation_metrics.py`.
- Conduct bias audits and propose improvements.

**Deliverables:**
- A fully deployed LLM-based application accessible via a web API.
- Documentation outlining the model’s capabilities, performance, and ethical considerations.
- Presentation or report summarizing the learnings and outcomes.

---

## 💻 **Getting Started**

**Prerequisites:**
1. Python 3.8+  
2. Libraries: PyTorch, TensorFlow, Hugging Face Transformers, FastAPI, ONNX Runtime, FAISS  
3. Basic knowledge of Python and machine learning

**Installation:**
```bash
git clone https://github.com/your-username/llm-study-plan.git
cd llm-study-plan
pip install -r requirements.txt
```

---

## 🤝 **Contributing**

We welcome contributions! Feel free to open issues for suggestions or submit pull requests with improvements, additional scripts, or recommended resources.

---

## 📜 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🚀 **Let’s Learn Together!**

Embark on this journey to deepen your understanding of deep learning and large language models. Explore, experiment, and create. Let’s work together to unlock the potential of cutting-edge NLP.
