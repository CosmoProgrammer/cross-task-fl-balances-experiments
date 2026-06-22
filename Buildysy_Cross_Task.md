## **Cross-Task Federated Backbone Aggregation with Selective State Space Models for Building Energy Analytics** 

Bhanu Kumar Naman Srivastava Pandarasamy Arjunan Dept of Mathematics and Computing RBCCPS RBCCPS Indian Institute of Science Indian Institute of Science Indian Institute of Science Bengaluru, India Bengaluru, India Bengaluru, India bhanukumar@iisc.ac.in snaman@iisc.ac.in samy@iisc.ac.in 

## **Abstract** 

Large-scale building management relies on two key analytics capabilities: _load forecasting_ for scheduling and demand response, and _anomaly detection_ for identifying equipment faults and energy inefficiencies. In practice, meter data is siloed across different owners and regulatory entities, while edge gateways deploying these models operate under bandwidth constraints. Federated learning (FL) addresses the privacy concern, yet existing FL pipelines for building energy train separate models per task, duplicating communication and relearning the same temporal patterns. In parallel, selective state space models (SSMs) have shown strong long-range modelling with linear-time dynamics but remain largely unexplored in federated building-energy analytics. We present a unified framework with two complementary components: (i) _MambaMixer_ , a multiscale bidirectional selective-SSM architecture for building-energy time series, and (ii) a _cross-task federated_ strategy that shares a common SSM backbone across clients from multiple tasks while keeping task-specific heads within each task group. We evaluate on real buildings from ASHRAE and LEAD 1.0 under FedAvg and FedProx, and compare against single-task FL, local-only training, and centralised baselines (LSTM, LSTM-AE, Informer, MSD-Mixer, ANNAE). Combining MambaMixer with cross-task backbone sharing delivers accurate, privacy-preserving, communication-efficient building analytics while preserving task-specific specialisation, a practical path toward deployable edge intelligence under realistic bandwidth and privacy constraints. The code for the implemetaion can be found at https://github.com/iisc-edge/cross-task-fl-balances26 

## **Keywords** 

Federated Learning, State Space Models, Building Energy, Anomaly Detection, Time Series Forecasting 

## **ACM Reference Format:** 

Bhanu Kumar, Naman Srivastava, and Pandarasamy Arjunan. 2026. CrossTask Federated Backbone Aggregation with Selective State Space Models for Building Energy Analytics. In _ACM Sustainability Week 2026 (ACM Sustainability Week Companion ’26), June 22–25, 2026, Banff, AB, Canada._ ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3765611.3815362 

## **1 Introduction** 

Buildings account for nearly 40% of global energy consumption, with a significant fraction lost to inefficient operation, equipment faults, and suboptimal control [6]. Improving efficiency at scale requires _load forecasting_ and _anomaly detection_ running continuously on meter data . This data, however, is fragmented, while the gateways hosting analytics typically operate over bandwidthconstrained commercial links. Centralising it is often impractical for privacy, regulatory, and operational reasons. Federated learning (FL) [10] has therefore emerged as a natural fit for building energy, enabling collaborative training without raw-data exchange while distributing computation across low-power edge devices; prior work has applied FL to distributed load forecasting [1, 2] and to privacy-preserving energy anomaly detection with federated deep autoencoders [7]. 

Current FL pipelines, however, treat forecasting and anomaly detection as entirely separate problems, each running its own parallel training loop on the same meter streams. This duplicates communication on already-scarce edge bandwidth and relearns the daily/weekly seasonality, occupancy cycles, and weather-dependent trends that both tasks rely upon. Federated multi-task learning has been explored in general [14], yet existing building-energy FL studies remain largely task-specific. Meanwhile, selective state space models (SSMs) such as Mamba [3, 4] have reshaped long-horizon sequence modelling with linear-time, input-dependent dynamics, yet remain largely unexplored in federated building-energy analytics; related multi-scale designs like MSD-Mixer [15] exploit decomposition but rely on fixed MLP temporal mixing. 

In this work, we develop a unified federated framework that jointly supports both tasks through a shared temporal backbone, exploiting common structure while preserving task-specific specialisation. We introduce _MambaMixer_ , a multi-scale bidirectional selective-SSM architecture for building energy (Sec. 2), and a _crosstask federated backbone aggregation_ strategy that aggregates one SSM backbone across clients from both tasks and aggregates taskspecific heads within each task group (Sec. 3). We evaluate on real-world buildings from ASHRAE and LEAD 1.0 against taskisolated FL, local-only training, and centralised deeplearning baselines (Secs. 4-5). 

## **2 MambaMixer Architecture** 

This work is licensed under a Creative Commons Attribution 4.0 International License. _ACM Sustainability Week Companion ’26, Banff, AB, Canada_ © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2199-1/2026/06 https://doi.org/10.1145/3765611.3815362 

MambaMixer (Fig. 1) processes a univariate window x ∈ R _[𝐿]_[×][1] of _𝐿_ =128 hourly samples (≈ 5.3 days). 

**Multi-scale decomposition.** Each scale [15] lane uses patch size _𝑝𝑠_ ∈{24 _,_ 12 _,_ 6 _,_ 2 _,_ 1} corresponding to daily, half-day, six-hour, two-hour, and pointwise temporal bands.Let x denote the input 

**Figure 1: MambaMixer. The input window is patchified at five scales** _𝑝_ ∈{24 _,_ 12 _,_ 6 _,_ 2 _,_ 1} **. Each lane applies a Patch Encoder (blue), BiMamba selective-SSM mixer (indigo), FiLM Cross-Scale Gate (orange), and Patch Decoder (green); the bottom row expands these four blocks. The blue–indigo–orange–green blocks form the** _**shared backbone**_ **; each scale feeds a pink forecasting head and a purple anomaly head, routed by two AdaptiveScaleRouters into the forecast** yˆ **and the reconstruction** xˆ **.** 

signal and r _𝑠_ denote the residual at stage _𝑠_ , initialized as r1 = x, and let ˆc _𝑠_ be the component reconstructed by lane _𝑠_ . 

**==> picture [177 x 10] intentionally omitted <==**

so each lane explains a distinct frequency band. An auxiliary whiteness loss on r _𝑆_ +1 encourages the final residual to behave like noise. **Selective SSM + BiMamba.** Inside every lane the temporal mixer is a bidirectional selective SSM. A continuous SSM [4] is defined as _ℎ_[′] ( _𝑡_ )=A _ℎ_ ( _𝑡_ )+B _𝑢_ ( _𝑡_ ), _𝑦_ ( _𝑡_ )=C _ℎ_ ( _𝑡_ ); Mamba [3] makes the discretised dynamics _selective_ by letting B, C, and the step size Δ depend on the input: 

**==> picture [205 x 10] intentionally omitted <==**

Input-dependent Δ _𝑡_ lets the state _refresh_ on informative timesteps (occupancy transitions, demand spikes) and _carry forward_ through uninformative ones-a strong inductive bias for bursty energy signals. Inspired by the bidirectional Mamba blocks used in Vision Mamba [17], we adopt a bidirectional selectiveSSM scan that runs a forward and a reversed pass in parallel, BiMamba( _𝑢_ )=SSM→ ( _𝑢_ )+SSM← (rev _𝑢_ ), capturing non-causal context in _𝑂_ ( _𝐿_ ) time. 

**Cross-scale gating.** Coarser lanes summarise global context; finer lanes should interpret local fluctuations in light of that context. The Cross-Scale Gate (Fig. 1) realises this with Feature-wise Linear Modulation (FiLM) [12]: from the pooled coarser embedding c _𝑠_ −1 

it produces a gain/bias pair that affinely modulates the current embedding z _𝑠_ [12], 

**==> picture [192 x 10] intentionally omitted <==**

**Dual heads and routing.** Each scale feeds two light heads (Fig. 1): a _forecasting_ head (55K params) and an _anomaly reconstruction_ head (147K params). Two AdaptiveScaleRouters weight the per-scale predictions with an input-dependent softmax [15] _𝜶_ =softmax(MLP(Pool(x))) to produce the final 24-hour forecast yˆ and the 128-step reconstruction xˆ. All backbone blocks in Fig. 1 together form the _shared backbone 𝜽 𝐵_ (830K params); the two heads are the task-specific parameters _𝝓𝐹_ (55K), _𝝓𝐴_ (147K). This backbone/head split is exactly what the federated protocol in Sec. 3 exploits. 

## **3 Cross-Task Federated Backbone Aggregation** 

Let C _𝐹 ,_ C _𝐴_ be the disjoint forecasting and anomaly client groups with C = C _𝐹_ ∪C _𝐴_ , and let _𝑛𝑖_ be the local dataset size of client _𝑖_ . At round _𝑟_ we set 

**==> picture [224 x 23] intentionally omitted <==**

The first summation pools backbone updates across both tasks while heads remain task-specialised; at deployment, the per-round heads _𝝓[𝑅] 𝐹[,][ 𝝓] 𝐴[𝑅]_[are paired with the shared backbone] _[ 𝜽][𝑅] 𝐵_[to yield two] 

2 

**Algorithm 1:** Cross-Task Federated Backbone Aggregation 

**==> picture [242 x 303] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input:  Global rounds  𝑅 , local epochs  𝐸 , learning rate  𝜂 , FedProx coefficient<br>𝜇 ( 𝜇 =0 recovers FedAvg);<br>clients C = C 𝐹 ∪C 𝐴 with task  𝑡𝑖 ∈{ 𝐹,𝐴 } and local data D 𝑖 ;<br>MambaMixer split: shared backbone  𝜽 𝐵 ; task heads { 𝝓𝐹 , 𝝓𝐴 }.<br>Output: ( 𝜽 [𝑅] 𝐵 [,][ 𝝓][𝑅] 𝐹 [,][ 𝝓] 𝐴 [𝑅] [)][.]<br>1 Initialize  𝜽 𝐵 [0] [,][ 𝝓] [0] 𝐹 [,][ 𝝓] 𝐴 [0] [randomly]<br>2 for  𝑟 = 1  to  𝑅 // Global communication rounds  do<br>3 foreach  client 𝑖 ∈C  in parallel  do<br>4 Receive ( 𝜽 [𝑟] 𝐵 [−][1] , 𝝓 [𝑟] 𝑡𝑖 [−][1] ) from server<br>5 ( 𝜽 [𝑖] , 𝝓 [𝑖] ) ←( 𝜽 [𝑟] 𝐵 [−][1] , 𝝓 [𝑟] 𝑡𝑖 [−][1] )<br>6 for  𝑒 = 1  to  𝐸 do<br>7 foreach  mini-batch 𝑏 ∼D 𝑖 do<br>8 Ltask ←L 𝑡𝑖 ( 𝑏 ; 𝜽 [𝑖] , 𝝓 [𝑖] ) // MSE or recon.<br>9 Lprox ← [𝜇] 2 �∥ 𝜽 𝑖 − 𝜽 𝑟𝐵 −1 ∥ [2] + ∥ 𝝓 [𝑖] − 𝝓 [𝑟] 𝑡𝑖 [−][1] ∥ [2][�]<br>10 ( 𝜽 [𝑖] , 𝝓 [𝑖] ) ←( 𝜽 [𝑖] , 𝝓 [𝑖] ) − 𝜂 ∇(Ltask + Lprox )<br>11 Send ( 𝜽 [𝑖] , 𝝓 [𝑖] ) to server<br>12 𝜽 [𝑟] 𝐵 [←] [�] 𝑖 ∈C � 𝑗 ∈C 𝑛𝑖 [𝑛] 𝑗 [𝜽] [𝑖] // Cross-task backbone agg.<br>13 foreach  task 𝑡 ∈{ 𝐹,𝐴 } do<br>14 𝝓 [𝑟] 𝑡 [←] [�] 𝑖 ∈C 𝑡 � 𝑗 ∈C 𝑛𝑖𝑡 [𝑛] 𝑗 [𝝓][𝑖] // Within-task head agg.<br>15 return ( 𝜽 [𝑅] 𝐵 [,][ 𝝓][𝑅] 𝐹 [,][ 𝝓] 𝐴 [𝑅] [)]<br>(a) Energy Forecasting (b) Anomaly Detection<br>0.300 Cross-Task FL (ours)Single-Task FL 0.30 Cross-Task FL (ours) Single-Task FL<br>0.275 Local-Only diverges  0.25 Local-Only<br>0.250 0.20<br>0.225 0.15<br>0.200 0.10<br>1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10<br>Communication Round Communication Round<br>Anomaly Test MSE<br>Forecasting Test MSE<br>**----- End of picture text -----**<br>


**Figure 2: Test loss over** 10 **FL rounds. (a) Forecasting MSE: Local-Only diverges while FL methods converge. (b) Anomaly MSE: Cross-Task FL reaches the lowest reconstruction error.** 

models differing only in the attached head. Algorithm 1 details the full procedure with an optional FedProx [8] proximal term ( _𝜇_ =0 _._ 01) for heterogeneous clients. The benefit is that clients never need labels for both tasks, each building contributes to one task but its backbone benefits from the other. 

## **4 Experimental Setup** 

**Datasets.** _ASHRAE Great Energy Predictor (GEP) III_ [11] contains hourly electricity readings from 1 _,_ 448 buildings across 16 sources; we use a stratified subset of 50 buildings spanning education, office, retail, and public-assembly types. _LEAD 1.0_ [5] is an expertannotated anomaly detection benchmark derived from ASHRAE GEP III; we sample 50 buildings with anomaly rates ranging from ∼ 1%–6% (median ≈ 2%). 

**Preprocessing.** Building time series are chronologically ordered, with missing values filled using forward-fill (ASHRAE) or perbuilding median imputation (LEAD). ASHRAE buildings with mostly zero values or insufficient observations are removed. To reduce skew from extreme consumption spikes, we apply a log(1 + _𝑥_ ) transform. Sliding windows of length _𝐿_ =128 h and stride 24 h are extracted; forecasting predicts the next _𝐻_ =24 h, while anomaly detection reconstructs the same window. 

**Training Configuration.** Each dataset contributes 35 federated training clients and 15 unseen-building test clients; training clients are further split chronologically into 70/20/10% train/validation/test partitions. All FL settings use the same MambaMixer backbone and differ only in training strategy: **Local-Only** (no aggregation), **Single-Task FL** (FedAvg within FL-35 groups), **Cross-Task FL, FedAvg** (proposed, FL-70), **Cross-Task FL, FedProx** ( _𝜇_ =0 _._ 01, FL-70), and **Centralized** (pooled-data, non-private reference). Additional centralized baselines include LSTM, LSTMAE [9], Informer [16], MSD-Mixer [15], and ANN-AE. FL training uses _𝑅_ =10 communication rounds and _𝐸_ =5 local epochs; all models are optimized with AdamW (lr 10[−][3] , batch size 32) and gradient clipping at 1 _._ 0. Anomaly detection employs mask-and-reconstruct training with mask rate 0 _._ 15, and thresholds are selected using validation F1. 

## **5 Results and Discussion** 

Table 1 lists the results; the centralised MambaMixer outperforms the centralised baselines. Figure 2 shows per-round test loss. Crosstask sharing shows improvement in anomaly precision. Cross-Task FL reaches anomaly precision 0 _._ 992, 13 _._ 0 percentage points above single-task FL (0 _._ 862), and the best AUC-ROC of any federated method (0 _._ 804). It also attains the lowest forecasting MAPE (22 _._ 1%), below every condition, including the centralised MambaMixer (28 _._ 3%), indicating that the joint objective regularises the backbone especially well for low-load buildings. 

**Competitive with centralised training.** Cross-Task FL trails the centralised MambaMixer by only 1% in _𝑅_[2] and 3 _._ 4 points in AUC-ROC without exposing raw data. It also _beats_ the centralised LSTM on forecasting (0 _._ 937 vs. 0 _._ 934) and the centralised LSTM-AE on anomaly AUC-ROC (0 _._ 804 vs. 0 _._ 796). FedProx ( _𝜇_ =0 _._ 01) yields the best FL F1 (0 _._ 539) and best FL _𝑅_[2] (0 _._ 939), confirming the proximal term absorbs cross-task heterogeneity. 

**Role of aggregation and the SSM core.** Local-Only underperforms on every metric and its forecasting loss _increases_ over rounds (Fig. 2(a)), suggesting aggregation is essential. Replacing BiMamba with MLP mixing (MSD-Mixer) drops centralised AUC-ROC from 0 _._ 838 to 0 _._ 819 and precision from 0 _._ 945 to 0 _._ 799, this suggests that selective-SSM core, not the multi-scale skeleton alone, drives the anomaly gains. Cross-task FL incurs only a marginal 0 _._ 5% drop in _𝑅_[2] (0 _._ 937 vs. 0 _._ 942), consistent with typical multi-task regularization effects [13] 

**Edge-deployable communication footprint.** A dual-federation deployment uploads two 830K backbones per round (6 _._ 64 MB); our design uploads one backbone plus a task head (3 _._ 54–3 _._ 91 MB), a 41–47% reduction (depending on the attached head). 

**Limitations** This work serves as an early step toward understanding cross-task federated learning for building energy analytics. While Single-Task FL slightly outperforms our method on _𝑅_[2] and MSE, it underperforms by 6 _._ 9 MAPE points, suggesting weaker generalisation across diverse building profiles. Similarly, LocalOnly attains the highest Recall by aggressively flagging anomalies, but its low Precision (0 _._ 641) indicates a low-threshold artefact, also observed in ANN-AE (0 _._ 762/0 _._ 222 Precision/Recall). The study evaluates stratified subsets of only 50 buildings each from ASHRAE GEP III and LEAD 1 _._ 0, and each FL configuration is run with a 

3 

||**Method**<br>**Setting**|**Forecasting**<br>_𝑅_2 ↑<br>MSE↓<br>MAPE (%)↓|**Anomaly Detection**<br>F1↑<br>AUC-ROC↑<br>Precision↑<br>Recall↑|
|---|---|---|---|
||**MambaMixer under Local and Federated Settings**|||
||Local-Only<br>Local<br>0.925<br>0.253<br>24.8<br>0.488<br>0.761<br>0.641<br>**0.394**<br>Single-Task FL<br>FL-35<br>**0.942**<br>**0.197**<br>29.0<br>0.518<br>0.801<br>0.862<br>0.370<br>**Cross-Task FL, FedAvg**<br>FL-70<br>0.937<br>0.214<br>**22.1**<br>0.526<br>**0.804**<br>**0.992**<br>0.358<br>**Cross-Task FL, FedProx**<br>FL-70<br>0.939<br>0.206<br>27.7<br>**0.539**<br>0.791<br>0.988<br>0.371|||
||**Centralized Model**|||
||**_MambaMixer_**<br>_Central_<br>**_0.946_**<br>**_0.181_**<br>**_28.3_**<br>**_0.551_**<br>**_0.838_**<br>**_0.945_**<br>_0.389_|||
||LSTM / LSTM-AE<br>Central<br>0.934<br>0.224<br>52.0<br>0.540<br>0.796<br>0.917<br>0.383<br>Informer<br>Central<br>0.940<br>0.203<br>38.9<br>-<br>-<br>-<br>-<br>MSD-Mixer<br>Central<br>0.944<br>0.190<br>30.0<br>0.532<br>0.819<br>0.799<br>0.399<br>ANN-AE<br>Central<br>-<br>-<br>-<br>0.343<br>0.645<br>0.222<br>**0.762**|||



**Table 1: Results on** 100 **buildings (** 15 **unseen-building test clients per task). The upper block reports results for the proposed model under local and federated training settings, while the lower block reports centralized baselines. Each metric is computed per held-out test building and then averaged across the** 15 **unseen-building test clients of the corresponding task. Values in bold indicate the best result on each metric within its category. “FL-** _𝑛_ **” denotes aggregation across** _𝑛_ **clients.** 

single fixed seed, limiting analysis of variability across splits, client sampling, and initialisation. In addition, experiments are restricted to _𝑅_ =10 communication rounds and _𝐸_ =5 local epochs; although convergence stabilises in Fig. 2, longer training may alter the relative ranking of methods. More extensive evaluations with repeated trials, larger cohorts, and per-use-type analyses remain important future work. 

## **6 Conclusion** 

We presented MambaMixer, a multi-scale bidirectional selective SSM, and cross-task federated backbone aggregation, a split FL topology that aggregates the SSM backbone across tasks while keeping heads task-local. On 100 real buildings it lifts anomaly precision by 13 percentage points over single-task FL, stays within 1% of centralised forecasting _𝑅_[2] , and cuts per-round communication by 41–47%. Future work will extend the topology to additional tasks and larger, more heterogeneous buildings. 

## **Acknowledgments** 

This work was supported by a research grant from the AI & Robotics Technology Park (ARTPARK) at the Indian Institute of Science. The authors also acknowledge the dataset providers for their efforts in making the data publicly available and maintaining high-quality resources for this study. 

## **References** 

- [1] Christopher Briggs, Zhong Fan, and Peter Andras. 2022. Federated learning for short-term residential load forecasting. _IEEE Open Access Journal of Power and Energy_ 9 (2022), 573–583. doi:10.1109/OAJPE.2022.3206220 

- [2] Mohammad Navid Fekri, Katarina Grolinger, and Syed Mir. 2022. Distributed load forecasting using smart meter data: Federated learning with recurrent neural networks. _International Journal of Electrical Power & Energy Systems_ 137 (2022), 107669. 

- [3] Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv preprint arXiv:2312.00752_ (2023). 

   - [5] Manoj Gulati and Pandarasamy Arjunan. 2022. LEAD1.0: A Large-scale Annotated Dataset for Energy Anomaly Detection in Commercial Buildings. In _Proceedings of the Thirteenth ACM International Conference on Future Energy Systems (e-Energy ’22)_ . ACM, 485–488. doi:10.1145/3538637.3539761 

   - [6] Tianzhen Hong, Zhe Wang, Xuan Luo, and Wanni Zhang. 2020. State-of-the-art on research and applications of machine learning in the building life cycle. _Energy and Buildings_ 212 (2020), 109831. 

   - [7] Bhanu Kumar, Naman Srivastava, Priyanka Nihalchandani, and Pandarasamy Arjunan. 2026. Privacy-Preserving Energy Anomaly Detection using Federated Deep Autoencoders. In _2026 18th International Conference on COMmunication Systems and NETworks (COMSNETS)_ . 1294–1296. doi:10.1109/COMSNETS67989. 2026.11418080 

   - [8] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. 2020. Federated optimization in heterogeneous networks. In _Proceedings of Machine Learning and Systems_ , Vol. 2. 429–450. 

   - [9] Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, and Gautam Shroff. 2016. LSTM-based encoder-decoder for multi-sensor anomaly detection. In _ICML 2016 Anomaly Detection Workshop_ . arXiv:1607.00148. 

   - [10] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-efficient learning of deep networks from decentralized data. In _Proceedings of the 20th International Conference on Artificial Intelligence and Statistics_ . PMLR, 1273–1282. 

   - [11] Clayton Miller, Pandarasamy Arjunan, Anjukan Kathirgamanathan, Chun Fu, Jonathan Roth, June Young Park, Chris Balbach, Krishnan Gowri, Zoltan Nagy, Anthony D. Fontanini, and Jeff Haberl. 2020. The ASHRAE Great Energy Predictor III competition: Overview and results. _Science and Technology for the Built Environment_ 26, 10 (2020), 1427–1447. 

   - [12] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. 2018. FiLM: Visual reasoning with a general conditioning layer. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 32. 

   - [13] Sebastian Ruder. 2017. An overview of multi-task learning in deep neural networks. _arXiv preprint arXiv:1706.05098_ (2017). 

   - [14] Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar. 2017. Federated multi-task learning. In _Advances in Neural Information Processing Systems_ , Vol. 30. 

   - [15] Shuhan Zhong, Sizhe Song, Weipeng Zhuo, Guanyao Li, Yang Liu, and S.-H. Gary Chan. 2024. A Multi-Scale Decomposition MLP-Mixer for Time Series Analysis. _Proceedings of the VLDB Endowment_ 17, 7 (2024), 1723–1736. doi:10.14778/3654621. 3654637 

   - [16] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. 2021. Informer: Beyond efficient transformer for long sequence time-series forecasting. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , Vol. 35. 11106–11115. 

   - [17] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. 2024. Vision Mamba: Efficient visual representation learning with bidirectional state space model. _arXiv preprint arXiv:2401.09417_ (2024). 

- [4] Albert Gu, Karan Goel, and Christopher Ré. 2022. Efficiently modeling long sequences with structured state spaces. In _International Conference on Learning Representations_ . 

4 

