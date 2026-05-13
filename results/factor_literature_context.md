# Literature Context for Prognostic MOFA Factors

Purpose: manuscript-ready literature framing for the strongest survival-associated latent factors in the TCGA pan-cancer multi-omics analysis. This note contextualizes the factor biology, flags claims that should be updated before submission, and links each therapeutic interpretation to published or regulatory evidence.

Last updated: 2026-05-13

## High-Level Interpretation

The most compelling aspect of the factor analysis is that the survival-predictive latent axes do not simply recapitulate single genes. They organize known cancer biology into interpretable, therapeutically coherent programs: protease-driven invasion plus canonical driver mutations (Factor 8), immune checkpoint copy-number amplification (Factor 6), squamous lineage identity and NECTIN4 targeting (Factor 1), luminal de-differentiation and synthetic lethality (Factor 15), and a thyroid/IDH differentiation gradient (Factor 12). In the published literature, these axes map onto established hallmarks of tumor progression: extracellular matrix remodeling, immune evasion, lineage-state dependence, chromatin/SWI-SNF vulnerabilities, PI3K/AKT dependence, and differentiation-linked IDH biology.

Two claims should be updated in the current manuscript draft:

1. **MRTX1133 status:** the ClinicalTrials.gov record for MRTX1133 in KRAS G12D-mutant solid tumors now lists the study as **terminated before phase 2 because of formulation challenges**. Describe MRTX1133 as a clinically tested investigational KRAS G12D inhibitor, not as an active phase I/II program unless a newer successor compound is being discussed.
2. **IDH glioma approval:** **vorasidenib**, not ivosidenib, is the FDA-approved IDH-targeted therapy for grade 2 IDH-mutant astrocytoma/oligodendroglioma. Ivosidenib remains FDA-approved in IDH1-mutant AML, cholangiocarcinoma, and relapsed/refractory MDS.

## 4.1 Factor 8: Protease-Driven Invasion and Pan-Driver Mutation Axis

Factor 8 is best framed as an invasion/proteolysis axis coupled to classical driver-gene instability. The expression loadings on **PRSS3, KLK6, PI3, FERMT1, SERPINB5, FGFBP1, and IL1A** are consistent with a literature in which tumor cells and stromal cells reshape the extracellular matrix, alter adhesion, release growth factors, and create motility-permissive niches. Broad reviews of tumor ECM remodeling emphasize that malignant progression involves deposition, biochemical modification, and degradation of matrix proteins, supporting growth, migration, and metastasis. More specifically, extracellular proteases, including serine proteases, can degrade basement membrane and interstitial barriers and can also alter signaling through cleavage of receptors, cytokines, and adhesion proteins.

The presence of **KLK6** is especially notable. Published colorectal cancer studies report that KLK6 expression is associated with aggressive cellular behavior, invasion, metastasis, and poor outcome. One lymph-node study found that KLK6 positivity marked aggressive disseminated tumor cells and worse recurrence/cancer-death risk. This makes the Factor 8 protease signature more than an annotation artifact: it fits a known invasive phenotype.

The mutation component of Factor 8, including **TP53, KRAS, APC, IDH1, ATRX, KMT2D, and BRAF**, places the protease program in a classical pan-cancer driver context. That pairing is biologically plausible: driver-mutated tumors often acquire matrix-remodeling, inflammatory, and survival programs that enable dissemination and therapy resistance.

At the CNV level, **GAS6** provides a strong literature bridge. GAS6 activates AXL, and the GAS6/AXL pathway is repeatedly linked to invasion, survival, angiogenesis, immune evasion, epithelial-to-mesenchymal transition, and drug resistance. Meta-analytic evidence also links high AXL expression to shorter overall and disease-free survival across solid tumors.

Therapeutic framing should be layered by maturity:

- **High clinical maturity:** KRAS G12C inhibitors are FDA-approved in specific settings. Sotorasib received accelerated approval for previously treated KRAS G12C-mutant NSCLC in 2021, and adagrasib received accelerated approval in 2022.
- **Investigational / update needed:** MRTX1133 is a KRAS G12D inhibitor with a first-in-human trial record, but that trial is now listed as terminated prior to phase 2. Use cautious language such as "clinical proof-of-concept programs for KRAS G12D inhibition have begun, but this specific program was terminated."
- **Mechanistically strong, clinically evolving:** GAS6/AXL inhibitors such as bemcentinib and multi-kinase inhibitors with AXL activity are rational in AXL-high contexts, but the predictive biomarker strategy is less mature than KRAS G12C.
- **Exploratory:** serine protease inhibition, anti-KLK6 strategies, and camostat/nafamostat-like protease blockade are biologically coherent but should be presented as preclinical/translational hypotheses unless tumor-specific clinical evidence is added.

Suggested manuscript language:

> Factor 8 links a protease-rich invasive state with canonical pan-cancer driver alterations. This aligns with a broad literature showing that extracellular proteolysis and ECM remodeling facilitate tumor-cell migration, stromal reprogramming, immune escape, and metastatic progression. The prominence of KLK6 further supports an invasion-associated interpretation, as KLK6 has been linked to poor prognosis, invasion, and metastatic dissemination in colorectal cancer models and patient cohorts. The concurrent GAS6 CNV signal nominates the GAS6/AXL survival and immune-evasion pathway, connecting Factor 8 to a therapeutically tractable axis of invasion, resistance, and microenvironmental suppression.

## 4.2 Factor 6: The 9p24.1 Immune Checkpoint Amplicon

Factor 6 has the strongest one-to-one mapping to a well-established literature-defined genomic lesion. Co-amplification of **CD274/PD-L1, PDCD1LG2/PD-L2, and JAK2** at **9p24.1** is a defining alteration in classical Hodgkin lymphoma and primary mediastinal large B-cell lymphoma. The foundational Blood study showed that 9p24.1 amplification increases PD-1 ligand gene dosage and that JAK2 amplification further induces PD-1 ligand transcription through JAK/STAT signaling. This provides a direct mechanistic explanation for why the factor loads on all three genes as a coherent CNV program.

The pan-cancer extension is also supported. Studies have identified 9p24.1 gains/amplifications in DLBCL, triple-negative breast cancer, glioblastoma, colorectal cancer, and other solid tumors, with increased expression of PD-L1, PD-L2, and JAK2 and generally adverse prognosis in amplified cohorts. A JAMA Oncology pan-solid-tumor analysis described PD-L1 amplification as uncommon but potentially predictive of checkpoint blockade sensitivity in selected solid tumors.

Therapeutically, the checkpoint implication is strong but should be written carefully:

- The most mature setting is lymphoma, especially classical Hodgkin lymphoma, where 9p24.1 copy-number alterations are a major biological basis for PD-1 blockade sensitivity.
- In solid tumors, CD274/PDCD1LG2 amplification is rarer, and evidence is more heterogeneous, but the alteration is a plausible biomarker for immune checkpoint blockade enrichment.
- JAK2 co-amplification supports the concept of JAK/STAT-driven PD-L1 transcription, but combining JAK inhibition with PD-1 blockade needs careful context because JAK signaling can also be required for interferon response and antitumor immunity. Phrase this as a testable combination hypothesis rather than a settled therapeutic rule.

Suggested manuscript language:

> Factor 6 recovers the 9p24.1 immune checkpoint amplicon as a survival-associated pan-cancer latent factor. The joint CNV signal over CD274, PDCD1LG2, and JAK2 mirrors the canonical lesion in classical Hodgkin lymphoma and primary mediastinal large B-cell lymphoma, where 9p24.1 amplification increases PD-1 ligand dosage and JAK2 activity further induces ligand transcription. Its appearance in a pan-cancer factor model suggests that a lymphoma-defined immune-evasion mechanism may also mark a smaller but clinically meaningful subset of solid tumors. This supports evaluating Factor 6-high tumors for PD-L1/PD-L2/JAK2 copy gain, PD-L1 expression, immune contexture, and checkpoint-inhibitor sensitivity.

## 4.3 Factor 1: Squamous Lineage Program and NECTIN4 Targeting

Factor 1 is most naturally interpreted as a **basal/squamous differentiation state**. The expression signature (**TP63, KRT16, KRT17, KRT15, NECTIN4, GJB2, SFN**) fits p63-governed squamous lineage biology across head and neck, lung squamous, cervical, esophageal, and urothelial-like epithelial contexts. The CNV gains in **DCUN1D1** also fit the squamous literature: 3q amplification is a recurrent feature of squamous cancers, and DCUN1D1/SCCRO has been studied as an oncogenic neddylation-pathway component in squamous tumors.

The most actionable part is **NECTIN4**. Enfortumab vedotin is an FDA-approved Nectin-4-directed antibody-drug conjugate in urothelial carcinoma, with regular approval in 2021 after earlier accelerated approval and first-line combination approval with pembrolizumab in 2023. The relevance beyond urothelial cancer is no longer purely speculative: Nectin-4 has been reported as widely expressed in HNSCC, and a phase II study of enfortumab vedotin in previously treated advanced head and neck cancer has been published.

Suggested manuscript language:

> Factor 1 captures a p63/basal-squamous lineage program with direct therapeutic implications through NECTIN4. The factor links canonical squamous markers with NECTIN4 expression, suggesting that a lineage-state score may identify tumors beyond urothelial cancer that retain vulnerability to Nectin-4-directed ADC therapy. This is supported by reports of broad Nectin-4 expression in HNSCC and by clinical testing of enfortumab vedotin in advanced head and neck cancer. Thus, Factor 1 can be framed as both a lineage classifier and a candidate ADC-sensitivity axis.

## 4.4 Factor 15: Luminal De-Differentiation and Synthetic Lethality

Factor 15 appears to describe loss of luminal epithelial differentiation, with negative loadings for **GATA3, MUC1, TFF1, SCGB2A2, CA12, and PRLR** and loadings involving **PTEN, ARID1A, KMT2C/KMT2D, and TP53**. This interpretation is consistent with endocrine-resistance and de-differentiation states in breast, prostate, and gynecologic cancers, but the strongest literature context is synthetic lethality and pathway dependence:

- **ARID1A loss and EZH2 inhibition:** ARID1A-mutant cancers show a synthetic-lethal dependence on EZH2 methyltransferase activity in preclinical studies. Tazemetostat is FDA-approved in other settings and is being tested in SWI/SNF-altered solid tumors.
- **PTEN loss and AKT dependence:** capivasertib plus fulvestrant is FDA-approved for HR-positive/HER2-negative advanced breast cancer with PIK3CA/AKT1/PTEN alterations after endocrine progression. The FDA approval was based on CAPItello-291, where altered tumors had median PFS of 7.3 months versus 3.1 months with placebo-fulvestrant.
- **KMT2C/KMT2D loss and PRMT5:** the PRMT5 connection is plausible but should be presented more cautiously than ARID1A/EZH2 and PTEN/AKT unless a specific KMT2C/KMT2D dependency paper is cited in the final manuscript.

Suggested manuscript language:

> Factor 15 places poor outcome within a de-differentiated luminal-state-loss framework. The negative loading of luminal epithelial markers suggests that tumors scoring adversely on this factor may have exited hormone-responsive epithelial identity and acquired chromatin- and PI3K/AKT-linked vulnerabilities. Among the therapeutic hypotheses, PTEN/PI3K/AKT pathway targeting has the strongest immediate clinical support because capivasertib plus fulvestrant is FDA-approved for HR-positive/HER2-negative advanced breast cancer with PIK3CA/AKT1/PTEN alterations. ARID1A loss provides a second strong rationale through EZH2 synthetic lethality, although translation in solid tumors remains under active clinical investigation.

## 4.5 Factor 12: Thyroid/IDH Differentiation Gradient

Factor 12 is best framed as a differentiation/prognosis gradient rather than a single pathway. The high-loading thyroid differentiation genes (**DUOX2, TG, TPO**) align with well-differentiated thyroid identity, while the IDH/glioma/RCC-associated mutation pattern points toward more aggressive dedifferentiated or non-thyroid contexts.

Therapeutic wording should be updated:

- **BRAF V600E in thyroid cancer:** BRAF V600E is common in papillary thyroid cancer and is clinically actionable in selected thyroid cancer contexts, but its prognosis is context-dependent and should not be universally described as "better survival."
- **IDH therapy:** ivosidenib is FDA-approved for IDH1-mutant AML, cholangiocarcinoma, and relapsed/refractory MDS. The FDA-approved IDH-targeted therapy for grade 2 IDH-mutant glioma is **vorasidenib**, an IDH1/IDH2 inhibitor approved in 2024 based on the INDIGO trial.
- **GPC5/GPC6:** glypicans are credible emerging cell-surface therapeutic targets, but most clinical development has focused on GPC2 and GPC3. For GPC5/GPC6, use "emerging glypican-family targets" rather than implying mature ADC/CAR-T validation.
- **TMPRSS2:** the TMPRSS2:ERG fusion occurs in roughly 40-70% of prostate cancers depending on cohort and assay, with many cohorts around 50%. Camostat/nafamostat should be discussed as protease-inhibition hypotheses, not established TMPRSS2:ERG-directed prostate cancer therapies.

Suggested manuscript language:

> Factor 12 captures a differentiation gradient spanning thyroid-lineage identity and IDH-associated tumor biology. The thyroid marker loadings support a well-differentiated epithelial state, while the opposing IDH/CIC/ATRX/PBRM1 pattern suggests transition toward glioma/RCC-like aggressive molecular contexts. This factor should be interpreted as a lineage/differentiation axis with therapeutic implications that differ by tumor type: BRAF-directed therapy in selected BRAF-mutant thyroid cancers, IDH inhibition with FDA-approved ivosidenib in AML/cholangiocarcinoma/MDS and vorasidenib in grade 2 IDH-mutant glioma, and exploratory evaluation of glypican-family surface targets where GPC5/GPC6 amplification is present.

## Suggested Cross-Factor Discussion Paragraph

Taken together, the top prognostic factors define a clinically interpretable map of pan-cancer survival biology. Factor 8 links extracellular proteolysis and invasion to canonical driver mutations and GAS6/AXL signaling; Factor 6 isolates the 9p24.1 PD-L1/PD-L2/JAK2 immune-evasion amplicon; Factor 1 captures basal/squamous lineage identity with NECTIN4-directed ADC implications; Factor 15 marks luminal-state loss and chromatin/PI3K synthetic-lethal vulnerabilities; and Factor 12 spans differentiated thyroid identity and IDH-associated tumor biology. The convergence between machine-learned latent factors and published cancer mechanisms supports the biological validity of the modeling framework. However, therapeutic interpretation should remain tiered: some links are clinically mature (KRAS G12C, PD-1 blockade in 9p24.1-altered lymphomas, enfortumab vedotin in urothelial cancer, capivasertib in PTEN/PIK3CA/AKT1-altered HR-positive breast cancer, vorasidenib in IDH-mutant grade 2 glioma), whereas others remain translational hypotheses requiring cancer-type-specific validation.

## Claim Strength Table

| Factor | Claim | Literature support | Suggested confidence |
|---|---|---:|---|
| Factor 8 | Protease/ECM invasion biology | Strong general support; gene-specific support for KLK6; translational for PRSS3/PI3 | Medium-high |
| Factor 8 | GAS6/AXL survival and immune-evasion axis | Strong review/meta-analysis support | Medium-high |
| Factor 8 | KRAS G12C therapeutic actionability | FDA-approved in defined KRAS G12C settings | High |
| Factor 8 | KRAS G12D MRTX1133 clinical actionability | Trial existed but is now terminated | Low-medium |
| Factor 6 | 9p24.1 CD274/PDCD1LG2/JAK2 amplicon | Very strong in cHL/PMBCL; growing pan-cancer evidence | High |
| Factor 6 | Checkpoint sensitivity in 9p24.1-amplified tumors | Strong in cHL; suggestive in selected solid tumors | Medium-high |
| Factor 1 | Squamous lineage/p63 program | Strong biological fit | High |
| Factor 1 | NECTIN4 ADC relevance beyond urothelial cancer | FDA-approved in UC; phase II HNC evidence | Medium-high |
| Factor 15 | ARID1A/EZH2 synthetic lethality | Strong preclinical; clinical translation evolving | Medium |
| Factor 15 | PTEN/AKT targeting with capivasertib | FDA-approved in HR+/HER2- breast cancer with pathway alterations | High in that indication |
| Factor 12 | IDH inhibitor relevance | High, but assign correct agent by disease context | High |
| Factor 12 | GPC5/GPC6 ADC/CAR-T targeting | Glypican family support, but GPC5/6-specific translation less mature | Low-medium |

## References And Source Links

### ECM, proteases, and KLK6

- Winkler J, Abisoye-Ogunniyan A, Metcalf KJ, Werb Z. Concepts of extracellular matrix remodelling in tumour progression and metastasis. *Nat Commun*. 2020. https://pubmed.ncbi.nlm.nih.gov/33037194/
- Martin CE, List K. Cell surface-anchored serine proteases in cancer progression and metastasis. *Cancer Metastasis Rev*. 2019. https://pubmed.ncbi.nlm.nih.gov/31529338/
- Goel A, Chauhan SS. Role of proteases in tumor invasion and metastasis. *Indian J Exp Biol*. 1997. https://pubmed.ncbi.nlm.nih.gov/9357157/
- Kalinska M, et al. Kallikrein-related peptidase 6 as a contributor toward an aggressive cancer cell phenotype. *Cancers*. 2022. https://pubmed.ncbi.nlm.nih.gov/35883559/
- Ohlsson L, et al. Lymph node tissue kallikrein-related peptidase 6 mRNA: a progression marker for colorectal cancer. *Br J Cancer*. 2012. https://pubmed.ncbi.nlm.nih.gov/22699826/

### 9p24.1 immune checkpoint amplicon

- Green MR, et al. Integrative analysis reveals selective 9p24.1 amplification, increased PD-1 ligand expression, and further induction via JAK2 in Hodgkin lymphoma and primary mediastinal large B-cell lymphoma. *Blood*. 2010. https://pubmed.ncbi.nlm.nih.gov/20628145/
- Barrett MT, et al. Genomic amplification of 9p24.1 targeting JAK2, PD-L1, and PD-L2 is enriched in high-risk triple negative breast cancer. *Oncotarget*. 2015. https://pubmed.ncbi.nlm.nih.gov/26317899/
- Goodman AM, et al. Prevalence of PDL1 amplification and preliminary response to immune checkpoint blockade in solid tumors. *JAMA Oncol*. 2018. https://jamanetwork.com/journals/jamaoncology/fullarticle/2684636
- Maurer MJ, et al. Amplification of 9p24.1 in diffuse large B-cell lymphoma identifies a unique subset. *Blood Cancer J*. 2019. https://www.nature.com/articles/s41408-019-0233-5

### GAS6/AXL

- Myers KV, Amend SR, Pienta KJ. Gas6/Axl signaling pathway in the tumor immune microenvironment. *Cancers*. 2020. https://pubmed.ncbi.nlm.nih.gov/32660000/
- Zhu C, et al. Therapeutic targeting of the Gas6/Axl signaling pathway in cancer. *Mol Cancer*. 2021. https://pubmed.ncbi.nlm.nih.gov/34576116/
- Zhang S, et al. The prognostic role of Gas6/Axl axis in solid malignancies: a meta-analysis and literature review. *Onco Targets Ther*. 2018. https://pubmed.ncbi.nlm.nih.gov/29416351/

### KRAS and TP53 therapeutics

- FDA. Sotorasib accelerated approval for KRAS G12C-mutated NSCLC. 2021. https://www.fda.gov/drugs/resources-information-approved-drugs/fda-grants-accelerated-approval-sotorasib-kras-g12c-mutated-nsclc
- FDA. Adagrasib accelerated approval for KRAS G12C-mutated NSCLC. 2022. https://www.fda.gov/drugs/resources-information-approved-drugs/fda-grants-accelerated-approval-adagrasib-kras-g12c-mutated-nsclc
- ClinicalTrials.gov. MRTX1133 in advanced solid tumors harboring KRAS G12D mutation, NCT05737706. https://www.clinicaltrials.gov/study/NCT05737706
- Sallman DA, et al. Eprenetapopt and azacitidine in TP53-mutant MDS. *J Clin Oncol*. 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8099410/
- Aprea Therapeutics. Phase 3 eprenetapopt trial in TP53-mutant MDS failed primary endpoint. 2020. https://ir.aprea.com/news-releases/news-release-details/aprea-therapeutics-announces-results-primary-endpoint-phase-3

### NECTIN4 and squamous/urothelial targeting

- FDA. Enfortumab vedotin regular approval in locally advanced/metastatic urothelial cancer. 2021. https://www.fda.gov/drugs/resources-information-approved-drugs/fda-grants-regular-approval-enfortumab-vedotin-ejfv-locally-advanced-or-metastatic-urothelial-cancer
- FDA. Enfortumab vedotin plus pembrolizumab approval in locally advanced/metastatic urothelial cancer. 2023. https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-enfortumab-vedotin-ejfv-pembrolizumab-locally-advanced-or-metastatic-urothelial-cancer
- Sanders C, et al. Nectin-4 is widely expressed in head and neck squamous cell carcinoma. *Oncotarget*. 2022. https://www.oncotarget.com/article/28299/text/
- Swiecicki PL, et al. Phase II trial of enfortumab vedotin in previously treated advanced head and neck cancer. *J Clin Oncol*. 2025. https://pubmed.ncbi.nlm.nih.gov/39481054/
- Hashimoto H, et al. Nectin-4: a novel therapeutic target for skin cancers. *Curr Treat Options Oncol*. 2022. https://pubmed.ncbi.nlm.nih.gov/35312963/

### ARID1A/EZH2, PTEN/AKT, and luminal de-differentiation therapeutics

- Bitler BG, et al. Synthetic lethality by targeting EZH2 methyltransferase activity in ARID1A-mutated cancers. *Nat Med*. 2015. https://pubmed.ncbi.nlm.nih.gov/25686104/
- FDA. Capivasertib with fulvestrant approval for HR+/HER2- breast cancer with PIK3CA/AKT1/PTEN alterations. 2023. https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-capivasertib-fulvestrant-breast-cancer
- Turner NC, et al. Capivasertib in hormone receptor-positive advanced breast cancer. *N Engl J Med*. 2023. https://pubmed.ncbi.nlm.nih.gov/37256976/

### IDH, glypicans, and prostate fusion context

- FDA. Ivosidenib approval for IDH1-mutant cholangiocarcinoma. 2021. https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-ivosidenib-advanced-or-metastatic-cholangiocarcinoma
- FDA. Ivosidenib approval for newly diagnosed AML with IDH1 mutation. 2019. https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-ivosidenib-first-line-treatment-aml-idh1-mutation
- FDA. Ivosidenib approval for relapsed/refractory MDS with IDH1 mutation. 2023. https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-ivosidenib-myelodysplastic-syndromes
- FDA. Drug Trials Snapshot: Vorasidenib for grade 2 IDH-mutant astrocytoma/oligodendroglioma. 2024. https://www.fda.gov/drugs/drug-approvals-and-databases/drug-trials-snapshots-voranigo
- Li N, Gao W, Zhang YF, Ho M. Glypicans as cancer therapeutic targets. *Trends Cancer*. 2018. https://pubmed.ncbi.nlm.nih.gov/30352677/
- Clark JP, Cooper CS. Common gene rearrangements in prostate cancer. *Cancer Lett*. 2010. https://pmc.ncbi.nlm.nih.gov/articles/PMC4874145/

