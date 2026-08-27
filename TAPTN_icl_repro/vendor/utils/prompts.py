import sys
import os
sys.path.append("./")
from utils.load_products import products_keys_list


def generate_arxiv_prompts(include_options, arxiv_natural_lang_mapping):
    
    keys=arxiv_natural_lang_mapping.keys()
    keys=[f'{key} ({arxiv_natural_lang_mapping[key]})' for key in ['cs.GT','cs.MA','cs.RO','cs.NE','cs.IR','cs.SI','cs.CY']]
    #keys=[f'{key} ({arxiv_natural_lang_mapping[key]})' for key in ['cs.RO','cs.CL','cs.AI','cs.LG']]
    if include_options:
        f"Predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper based on the titles and abstracts of the paper itself as well as its references and citations. Choose from the following categories:{', '.join(keys)}. The predicted sub-category should be in the format 'cs.XX'."
        f"Further revise the initial categorization for the paper. Choose the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper from the following categories:{', '.join(keys)}.The predicted sub-category should be in the format 'cs.XX'.\n\n"+"{}\n\nNow, apply this method to the following paper. The predicted sub-category should be in the format 'cs.XX'. If multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n"
        arxiv_prompts = {
            'subcategory': f"Further revise the initial categorization for the paper. Choose the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper from the following categories:{', '.join(keys)}.The predicted sub-category should be in the format 'cs.XX'.\n\n"+"{}\n\nNow, apply this method to the following paper. The predicted sub-category should be in the format 'cs.XX'. If multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n",
            'identifier': f"Please predict the most appropriate original arxiv identifier for the paper. Your answer should be chosen from {', '.join([key.lower() for key in arxiv_natural_lang_mapping.keys()])}. The predicted arxiv identifier should be in the format 'arxiv cs.xx'.",
            'natural language': "Please predict the 2 most appropriate category for the paper based on the titles and abstracts of the paper itself as well as its references and citations. Your answer should be chosen from {}.".format(', '.join(['"{}"'.format(arxiv_natural_lang_mapping[key]) for key in arxiv_natural_lang_mapping.keys()]))
        }
    else:
        "Please predict the most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format 'cs.XX'."
        "Please predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper based on the titles and abstracts of the paper itself as well as its references and citations. The predicted sub-category should be in the format 'cs.XX'."
        f"Further revise the initial categorization for the paper. Predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format 'cs.XX'.\n\n"+"{}\n\nNow, apply this method to the following paper. The predicted sub-category should be in the format 'cs.XX'. If multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n"
        "Further refine the initial categorization for the paper. Predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format 'cs.XX'.\n\n{}\n\nNow, apply this method to the following paper.\n\n"
        arxiv_prompts = {
            'subcategory':  "Further refine the initial categorization for the paper. Predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format 'cs.XX'.\n\n{}\n\nNow, apply this method to the following paper.\n\n",
            'identifier': "Please predict the most appropriate original arxiv identifier for the paper. The predicted arxiv identifier should be in the format 'arxiv cs.xx'.",
            'natural language': "Please predict the most appropriate category for the paper. Your answer should be chosen from {}.".format(', '.join(['"{}"'.format(arxiv_natural_lang_mapping[key]) for key in arxiv_natural_lang_mapping.keys()]))
        }
    return arxiv_prompts

refining3="""Choosing the most appropriate category for a paper based on the titles and abstracts of the paper and its references and citations involves a process of identifying key themes, methods, and subject areas. Here is a structured approach to help you categorize a paper:

1. **Understand the Categories**: First, familiarize yourself with the predefined list of categories available. Understand what each category entails, including the typical methodologies, subject areas, and scopes covered by each.

2. **Analyze the Paper's Abstract and Title**:
    - **Keywords and Phrases**: Identify key terms and phrases in the title and abstract. These often indicate the central themes and the discipline.
    - **Objective and Approach**: Look for statements about the paper's main objectives and the methods used. This can give clues about whether the paper is theoretical, empirical, review-based, etc.
    - **Subject Area**: Determine which area of study the paper belongs to based on the problems addressed and the context provided.
    
3. **Examine References**:
    - **Reference Sources**: Review the titles and abstracts of the cited papers. Papers often cite sources within the same or a closely related field.
    - **Patterns in Citations**: Note any recurring themes or predominant disciplines in the citations. This can indicate the community and academic discourse the paper is engaging with.

4. **Check Citations to the Paper**:
    - **Citing Papers' Focus**: Look at what aspects of the paper are being cited by other authors and in what context. This might provide insights into the paper's contributions and relevance in specific fields.
    
5. **Synthesize the Information**:
    - **Majority Rule**: If the majority of references and citations belong to a particular category, there's a good chance the paper fits there too.
    - **Consistency Check**: Ensure the identified category aligns with the paper's objectives and methods.
    - **Broader Context**: Consider if the paper might be interdisciplinary and whether it should be categorized under a more general or specific field based on its breadth and focus.

6. **Make a Decision**:
    - **Best Fit**: Choose the category that best captures the essence of the paper based on the synthesis of the above steps.
    - **Fallback Option**: If unsure, lean towards a broader category that encompasses multiple aspects of the paper.
    
7. **Document the Reasoning**: It's useful to keep notes on why you categorized the paper in a certain way, especially if the decision was not straightforward. This helps in maintaining consistency when categorizing other papers.

8. **Final Check**: After final decision, review the reasoning process and final categorization carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.

This method relies heavily on critical thinking and a good grasp of the subject areas represented in your category list. It requires an analytical approach to text and the ability to discern patterns and themes from limited information.\n\n"""

# v2（2026-06，针对新一代模型修订）：原 refining3 由旧模型生成、无输出格式约束。
# 新模型推理更长、会显式枚举/排除所有类别名，导致下游答案抽取经常抓错类别
# （诊断：cora taptn1 新增错误中相当比例的推理结论正确但抽取结果错误）。
# 修订仅追加收尾格式要求，不改变推理方法本身。
refining3_v2 = refining3.rstrip() + """

IMPORTANT — Output format requirement: After your reasoning, you MUST end your entire response with a single plain-text line in exactly this format (no bold, no markdown, nothing after it):
Final answer: <most appropriate category>; <second most appropriate category>
Both categories must be copied verbatim from the given category list, ordered from most to least appropriate.\n\n"""

# ─────────────────────────────────────────────────────────────────────────────
# Cora Stage-1 思维干预（E1-k / E1-l）。两套均替代 refining3 注入第二槽，
# 通过环境变量 CORA_INTERVENE = {none|agnostic|era} 在 generate_system_prompt 中切换。
#   - agnostic：不考虑年代，仅强化“推理质量”脚手架（目标论文隔离 + 贡献产物 +
#               全类打分去平局 + 通用 NN/Prob 边界）。
#   - era    ：在同一脚手架上叠加“数据集时代分类法校准”——告知标注遵循
#               McCallum-2000 / NIPS-1990s 口径（NN 涵盖概率化连接主义模型；
#               Theory=CLT 涵盖规则/自动机可学习性；Probabilistic 限核心即概率模型）。
# 两者都保留与 refining3_v2 一致的收尾输出格式，确保下游答案抽取稳定。
# ─────────────────────────────────────────────────────────────────────────────
_CORA_OUTPUT_FMT = """

IMPORTANT — Output format requirement: After your reasoning, you MUST end your entire response with a single plain-text line in exactly this format (no bold, no markdown, nothing after it):
Final answer: <most appropriate category>; <second most appropriate category>
Both categories must be copied verbatim from the given category list, ordered from most to least appropriate.\n\n"""

cora_intervene_agnostic = """Use the following disciplined protocol to categorize the TARGET paper. This protocol corrects four common reasoning failures; apply every step explicitly.

1. **Isolate the target paper (anti citation-theme voting)**: The label belongs to the TARGET paper's own contribution, NOT to the dominant theme of its neighbors/references. Neighbors are weak context, not votes. A paper that cites or is cited by many works of category X is NOT thereby category X. First decide from the target's own title/abstract; use neighbors only to break a genuine tie.

2. **Classify by the contributed artifact, not by techniques used (anti method-vs-substrate confusion)**: Ask "What does this paper actually BUILD and EVALUATE as its central artifact/result?" Categorize by that artifact, not by tools it merely uses, borrows, analyzes, or compares against. Using/analyzing technique X does not make the paper an X paper.

3. **Score every class, then disprove the runner-up (anti tie inflation)**: Briefly rate the fit of ALL candidate categories. Do NOT assign equal top scores to two categories — you must produce a strict ordering. Take the top two, and write one sentence stating why the winner's defining criterion is met and the runner-up's is NOT.

4. **Resolve the Neural-Networks vs Probabilistic-Methods boundary (anti abstraction bias)**: Do not default to the "more fundamental/encompassing" category. If the paper's central artifact is a trained network of interconnected units / weights (even if trained or analyzed with probabilistic tools), it is Neural Networks. Choose Probabilistic Methods only when the central contribution is the probabilistic model/inference itself, with no neural network as the artifact.
""" + _CORA_OUTPUT_FMT

cora_intervene_era = """Use the following disciplined protocol to categorize the TARGET paper. CRITICAL CONTEXT: this category scheme is the historical Cora taxonomy (McCallum et al., 2000), which encodes late-1990s machine-learning community/venue boundaries (e.g., the NIPS "Neural Information Processing Systems" community). Classify by the conventions of THAT era, NOT by today's terminology. Apply every step explicitly.

1. **Isolate the target paper**: The label belongs to the TARGET paper's own contribution, NOT to the dominant theme of its neighbors/references. Decide from the target's own title/abstract first; use neighbors only to break a genuine tie. Citing/being-cited-by category X does not make the paper category X.

2. **Classify by the contributed artifact, not by techniques used**: Ask "What does this paper actually BUILD and EVALUATE as its central artifact?" Categorize by that artifact, not by tools it merely uses or analyzes.

3. **Apply the 1990s taxonomy conventions (anti modern-relabeling) — this is the most important step**:
   - **Neural Networks (1990s sense)** is the connectionist umbrella and INCLUDES probabilistic/Bayesian connectionist models: Gaussian processes, mixtures of experts, Helmholtz machines, Boltzmann machines, sigmoid belief networks, Bayesian neural networks, and EM-trained network models. Do NOT move these to Probabilistic Methods merely because they involve probability — in this taxonomy they are Neural Networks.
   - **Theory** here means Computational Learning Theory and INCLUDES PAC learning, VC dimension, sample/mistake-bound complexity, and learnability results — even when the objects learned are rules, concepts, decision lists, or automata. Do NOT relabel such papers as Rule Learning just because rules are being learned.
   - **Probabilistic Methods** is reserved for papers whose central contribution IS the probabilistic model or inference itself (e.g., graphical models, HMMs, Bayesian networks for inference) WITHOUT a neural network as the built artifact.
   - **Rule Learning** is for inducing explicit symbolic rules/logic programs as the artifact (e.g., ILP, association/decision rules), not for theoretical learnability of rules.

4. **Score every class, then disprove the runner-up**: Briefly rate ALL candidate categories, produce a strict ordering (no ties), and state in one sentence why the winner's era-criterion is met and the runner-up's is NOT.
""" + _CORA_OUTPUT_FMT

refining6 ="""Choosing the most appropriate category for a paper based on the titles and abstracts of the paper and its references and citations involves a process of identifying key themes, methods, and subject areas. Here is a structured approach to help you categorize a paper:

1. **Understand the Categories**: 
    - **Familiarize Yourself**: Understand each category in the list. Know the typical topics, methodologies, and scopes they cover.

2. **Analyze the Paper's Abstract and Title**:
    - **Identify Keywords**: Extract key terms and phrases that hint at the paper's primary focus.
    - **Determine Objective and Approach**: Look for descriptions of the paper's aims and methods.
    - **Contextual Clues**: Note any mentioned applications, theories, or subject areas.

3. **Examine References (Sources the Paper Cites)**:
    - **Discipline and Domain**: Identify the primary field of the references. Papers often cite works within the same or closely related disciplines.
    - **Recurring Themes**: Look for common themes, theories, or methodologies among the references.
    - **Influence and Foundation**: Understand the foundational work and theories the paper builds on. This provides context on its academic roots and primary field.

4. **Examine Citations (Sources Citing the Paper)**:
    - **Impact and Application**: Analyze how other papers are using the current paper. Are they building upon it, applying its findings, or challenging its methods?
    - **Context of Citations**: Identify the fields or categories where the paper's contributions are being recognized or utilized.
    - **Relevance and Scope**: Understand the broader impact and the specific aspects of the paper that are being cited.

5. **Synthesize the Information**:
    - **Consolidate Findings**: Combine insights from the abstract, title, references, and citations.
    - **Majority Rule for References**: If most references fall within a specific category, it likely indicates the paper's primary academic field.
    - **Check Consistency with Citations**: Ensure that the categories reflected in citations align with the reference-based preliminary category.
    - **Interdisciplinary Nature**: Consider if the paper spans multiple categories and decide whether it should be placed in a broader or more specific category.

6. **Make a Decision**:
    - **Select Best Fit**: Choose the category that best encapsulates the essence of the paper, considering both references and citations.
    - **Fallback Option**: Opt for a broader category if the paper's scope is interdisciplinary or if categorization remains unclear.

7. **Document the Reasoning**: 
    - **Record Decision Process**: Note the key points and reasons for selecting a particular category to ensure transparency and consistency in future categorizations.
    
8. **Final Check**: After final decision, review the reasoning process and final categorization carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.

This method relies heavily on critical thinking and a good grasp of the subject areas represented in your category list. It requires an analytical approach to text and the ability to discern patterns and themes from limited information.
"""
refining5="""To further revise the categorization of a paper based on the citation network and the initial categorizations, follow this structured method:

    1. **Examine the Citation Network**:
        - **Analyze Connections**: Look at how the paper is connected within the citation network. Identify whether it is primarily citing or being cited by papers within specific categories.
        - **Identify Influential Papers**: Determine which papers in the citation network are highly influential or frequently cited. These papers can often guide you towards the core category of the subject matter.

    2. **Compare Initial Categorizations**:
        - **Consistency Check**: Check if the initial categorizations of the papers within the citation network align with the initial categorization of the target paper. A strong alignment suggests a correct initial categorization.
        - **Majority Rule**: If the majority of the papers in the citation network belong to a particular category, this might indicate the central focus area for the target paper.

    3. **Review Reasoning for Categorizations**:
        - **Justifications**: Evaluate the reasons given for the initial categorizations of the papers in the citation network. Strong, well-articulated justifications can help validate the categories.
        - **Identify Common Themes**: Look for common themes in the reasoning provided. If similar reasons are repeatedly used for categorizing papers into a specific category, this strengthens the case for that category.

    4. **Cross-Referencing Themes**:
        - **Abstract and Title Analysis**: Re-examine the abstracts and titles of the target paper and the papers in its citation network. Look for shared keywords, phrases, and thematic overlaps.
        - **Methodologies and Approaches**: Compare the methodologies and approaches described in the abstracts. Similar methods often indicate similar categorical alignment.

    5. **Consider the Influence of Interdisciplinary Connections**:
        - **Broader Context**: Determine if the paper spans multiple disciplines. If it does, consider which categories are most relevant based on the depth and focus of the interdisciplinary connections.
        - **Primary vs. Secondary Categories**: If the paper is highly interdisciplinary, you might need to choose a primary category that best represents the core contribution and a secondary category for the supporting discipline.

    6. **Iterative Adjustment**:
        - **Re-Evaluate Initial Judgment**: Based on the analysis of the citation network and the comparisons made, re-evaluate the initial categorization.
        - **Revise if Necessary**: Adjust the category if the evidence from the citation network strongly supports a different categorization.

    7. **Final Decision**:
        - **Best Fit Category**: Choose the category that now seems to best capture the essence of the paper, considering the additional information from the citation network.
        - **Document the Revision**: Make notes on why the revision was made, including the influence of the citation network and any key papers that led to the change in categorization.

    8. **Final Review**: 
        - **Self-Check**: Review  the analysis process and the final categorization decision carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure that the analysis process is logically correct and fitting with the paper itself as well as its citation network, the final categorization decision aligns with the overall analysis and that the paper is placed where it best fits within the academic landscape.

By systematically analyzing the citation network and considering the broader context provided by the initial categorizations and reasons, you can refine and improve the judgment of the most appropriate categories for the paper. This approach ensures that the categorization is robust, well-justified, and reflective of the paper's true academic context."""
#dir cot rec6
# refining="Refining the initial categorization for the paper involves a few key steps that focus on reviewing, adjusting, and confirming the categories assigned. Here are the steps to further refine the categorization:\n\n1. **Cross-Verification**:\n     - Consistency Check**: Review the initial categories assigned to ensure consistency across similar papers. If papers with similar themes, methods, or citations are categorized differently, reassess why and consider harmonizing the categorization\n\n2. **Detailed Review of Borderline Cases**:\n    - Re-read Abstracts and Titles**: For papers that were difficult to categorize or ended up in broader or \"catch-all\" categories, revisit their abstracts and titles. Look for nuanced details that might have been overlooked.\n    - **Reassess References and Citations**: Take another look at the references and citations for these papers to see if there's a predominant category among these that could better fit the paper.\n\n3. **Feedback Loop**:\n    - **Adjustments Based on Themes**: If you notice recurring themes or methods emerging that weren't clearly categorized initially, consider adjusting the category definitions themselves or creating sub-categories to better accommodate these nuances.\n    - **Documentation**: Update your reasoning and any changes made to ensure that the categorization process remains transparent and reproducible.\n\n4. **Cluster Analysis**:\n    - **Similarity Clustering**: Clustering papers based on the similarity of their keywords, references, and cited papers could help identify natural groupings. Reassign categories based on these clusters.\n\n5. **Review Categories**:\n    - **Definition Clarity**: Ensure that each category is well-defined and distinct from others. Refine the definitions if there is overlap causing confusion in categorization.\n    - **Category Fit**: Reassess whether each paper's objectives, methodology, and subject matter align with the category's intended scope.\n\n6. **Iterative Refinement**:\n    - **Iterate the Process**: Sometimes, multiple rounds of review and adjustment may be necessary, especially as you refine category definitions and better understand the papers' contents.\n    - **Final Confirmation**: After adjustments, do a final review to confirm that each paper is placed in the most appropriate category according to the updated criteria.\n\nThe goal of these steps is to ensure that the categorization accurately reflects the papers' content and fits within the academic discourse they contribute to. This process is typically dynamic and may require several iterations to perfect, especially when dealing with a diverse and complex body of work."

#cotrec1,2
# refining="Refining the categorization of a paper after the initial assessment can lead to a more accurate and contextually appropriate classification. Given that you now have additional information about each paper's initial categorization and the reasons for these categorizations, along with the existing details of the abstract, title, references, and citations, you can use a more nuanced approach. Here's a step-by-step method to refine the categorization:\n\n1. **Review Initial Categorizations**:\n    - **Consistency Check**: Assess the initial categorization of the paper against its references and citations. Check if there are inconsistencies or mismatches in categorization.\n    - **Feedback Loop**: Use the reasons provided for initial categorizations to identify any systemic errors or biases that might have influenced the decisions. Look for any patterns that suggest misinterpretation of the category definitions.\n\n2. **Cross-Referencing**:\n    - **References and Citations Alignment**: Evaluate whether the references and citations of the paper are mostly categorized in the same or related fields. Misalignments might indicate a need to reconsider the paper's categorization.\n    - **Citation Network Analysis**: Look at the categorization of papers that cite your main paper and those that it cites. This can reveal how the community perceives the relevance and field of the paper.\n\n3. **Reassess Key Themes and Disciplines**:\n    - **Keyword Analysis**: Revisit the keywords in the abstract and title. Check if these keywords align well with the chosen category, especially in light of the new information from the initial categorization.\n    - **Discipline-Specific Terms**: Look for terms that are unique or especially significant to certain disciplines that might not have been adequately considered before.\n\n4. **Interdisciplinary Considerations**:\n    - **Boundary Spanning**: If the paper bridges multiple categories (evident from the diversity in the categorization of references and citations), consider whether an interdisciplinary category would be more appropriate.\n    - **Core vs. Peripheral**: Determine whether the paper's primary focus aligns with the core of a category or if it merely touches on multiple categories peripherally.\n\n5. **Engage with Anomalies**:\n    - **Outliers**: Identify any references or citations whose categorization significantly differs from the majority. Investigate whether these outliers suggest an overlooked aspect of the paper.\n    - **Anomalous Reasoning**: Review the reasoning for categorizing outlier papers and see if similar considerations might affect the categorization of your main paper.\n\n6. **Consensus Building**:\n    - **Synthesize Information**: Bring together insights from the analysis of keywords, themes, disciplines, and the categorization patterns of references and citations.\n    - **Refined Category Selection**: Choose a category that now appears most representative of the paper's focus and contributions, taking into account the broader context and interconnections revealed through the analysis.\n\n7. **Documentation and Adjustment**:\n    - **Record Changes**: Document any changes in categorization and the detailed reasons for these changes. This will be crucial for transparency and for future reference.\n    - **Iterative Review**: Consider this process as iterative. As more papers are categorized and as categorizations are refined, continue to reassess categories to maintain alignment and accuracy.\n\nThis refined approach ensures a deeper engagement with the material and a more thorough understanding of its placement within the academic landscape, enhancing the accuracy and relevance of the categorization."

#cotrec3
# refining="""To refine the initial categorization, you can follow these steps:

# 1. **Reassess Initial Reasons**: Review the reasons given for the initial categorization. Ensure they are solid and well-supported by the abstract, title, and other textual data.

# 2. **Cross-Referencing**: Check if the categories assigned to the references and citations are consistent with those of the main paper. Look for any discrepancies or patterns that might suggest a different category.

# 3. **Analyze Feedback**: If feedback is available from any review or audit process, incorporate this to correct any evident misclassifications.

# 4. **Comparative Analysis**: Compare the paper with other papers in the same initial category. Ensure it aligns well with them in terms of content, methodology, and discourse.

# 5. **Look for Overlaps**: Identify any interdisciplinary elements or overlaps with other categories, adjusting the category if necessary to better reflect the paper's scope.

# 6. **Update Reasons**: Amend the reasons for categorization based on any new insights or corrections, ensuring each decision is well-documented.

# By methodically applying these steps, you can enhance the accuracy of your categorization process."""

# refining="""To further refine the initial categorization, follow these steps:

# 1. **Review Initial Judgments**: Reassess the initial categorization and the reasons provided for each paper. Ensure that the reasons align closely with the chosen category.

# 2. **Cross-Check with References and Citations**: Compare the initial categorization of the paper with the categories of its references and citations. Check for consistency and thematic alignment across these related papers.

# 3. **Identify Anomalies**: Look for any inconsistencies or outliers in categorization among the paper, its references, and citations. Investigate why these discrepancies exist—whether due to interdisciplinary content, evolving research fields, or misinterpretation.

# 4. **Consolidate Feedback**: If any patterns of misalignment or frequent re-categorization are noted among related papers, consider whether the category definitions might need adjustment or if certain papers consistently fall between categories.

# 5. **Re-evaluate with Broader Scope**: Take into account the broader context of each paper's research area, especially for interdisciplinary works. Check if a different category might provide a better fit based on the extended analysis.

# 6. **Final Adjustment**: Make any necessary adjustments to the categorization based on the above analyses, ensuring that each paper is placed in the most relevant category according to its content and context within the field.

# By conducting this thorough review and adjustment process, the categorization can be refined to more accurately reflect the content and context of each paper."""


def generate_system_prompt(source, arxiv_style="subcategory", include_options=False, exlain=False, comfirm=False, options=None, use_instructions=True):
    """
    Generate a system prompt based on the given content type and source.
    
    Args:
    - content_type (str): Specifies the type of content (e.g., title, abstract, neighbors).
    - source (str): Specifies the data source (e.g., arxiv, cora, pubmed, product).
    - use_original_arxiv (bool, optional): If set to True, a special prompt for 'arxiv' is used.
    
    Returns:
    - str: Generated system prompt.
    """

    categories = {
        'cora': ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Computational Learning Theory", "Reinforcement Learning", "Probabilistic Methods"],
        'pubmed': ["Type 1 diabetes", "Type 2 diabetes", "Experimentally induced diabetes"]
    }
    if options:
        categories['cora'] = options
    arxiv_prompts = generate_arxiv_prompts(include_options, arxiv_natural_lang_mapping)
    
    prompts = {
        'arxiv': arxiv_prompts[arxiv_style],
        'cora': "Please predict the most appropriate category for the paper based on the titles and abstracts of the paper and its references and citations. Choose from the following categories:\n\n{}\n\n{}",
        'pubmed': "Please predict the most likely type of the paper. Your answer should be chosen from:\n\n{}",
        'product': "Please predict the most likely category of the target product from Amazon based on its name and description as well as the frequently purchased-together products. Your answer should be chosen from the list:\n\n{}\n\n{}\n\nNow, apply this approach to the following target product.\n\n"
    }
    if exlain:
        prompts['cora']='Please predict the 2 most appropriate categories for the paper. Choose from the following categories:\n\n{}'
        prompts['cora']="Please predict the 2 most appropriate categories for the paper based on the titles and abstracts of the paper and its references and citations. Choose from the following categories:\n\n{}\n\n{}"
    if not comfirm:
        prompts['cora']= "Please refine the initial categorization for the paper. Choose the most appropriate category for the paper from the following categories:\n\n{}\n\n{}"
        prompts['cora']= "Further revise the initial categorization for the paper. Choose the most appropriate category for the paper from the following categories:\n\n{}\n\n{}\n\nNow, apply this method to the following paper. If multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n"
        #prompts['cora']= "Further revise the categorization of the paper. Choose the most appropriate category for the paper from the following categories:\n\n{}"+"\n\n{}\n\nNow, apply this process to the following paper.\n\nIf multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n"
        prompts['cora']="Please predict the 2 most appropriate categories for the paper based on the titles and abstracts of the paper itself as well as its references and citations. Choose from the following categories:\n\n{}\n\n{}"
        #prompts['arxiv']="Please predict the 2 most appropriate categories for the paper based on the titles and abstracts of the paper itself as well as its references and citations. Choose from the following categories:\n\n{}\n\n{}"
        prompts['product']= "Further refine the initial categorization of the target product. Choose the most appropriate category for the target product from the following categories:\n\n{}\n\n{}\n\nNow, apply this approach to the following target product.\n\n"
        prompts['product']= "Refine the initial categorization of the target product after obtaining its initial category and reasoning. Predict the most appropriate category of the target product from Amazon based on its name, description, initial categirization and reasoning as well as the frequently purchased-together products. Your answer should be chosen from the list:\n\n{}\n\n{}\n\nNow, apply this approach to the following target product.\n\n"
        prompts['product']= "Refine the initial categorization of a target product from Amazon based on its name, description, initial categorization and reasons as well as these information of the frequently purchased-together items. Please think comprehensively from multiple perspectives, eliminate deviations in the definition of product categories, and discover and correct logical and factual errors in the initial classification. Choose the most appropriate category for the target product from the following categories:\n\n{}\n\n{}\n\nNow, integrate this process to the following target product to refine its initial categorization. Please think comprehensively from multiple perspectives, eliminate deviations in the definition of product categories, and discover and correct logical and factual errors in the initial classification.\n\n"
        prompts['product']= "As an expert, you need to revise the initial categorization of a target product from Amazon made by your rookie subordinate based on its name, description, initial categorization and reasons as well as this information of the frequently purchased-together items. Please think comprehensively from multiple perspectives, eliminate deviations in the definition of product categories, and discover and correct logical and factual errors in the initial classification. Choose the most appropriate category for the target product from the following categories:\n\n{}\n\n{}\n\nNow, apply this approach to the following target product.\n\n"
      #   prompts['product']= "Refine the initial categorization of a target product from Amazon based on its name, description, initial categorization and reasons as well as these information of the frequently purchased-together items. Choose the most appropriate category for the target product from the following categories:\n\n{}\n\nThink comprehensively from multiple perspectives, eliminate deviations in the understanding of product categories in the initial categorization, and discover and correct logical and factual errors in the initial classification.\n\n"
      #   prompts['product']= "Please predict the most likely category of the target product from Amazon based on its name, description, initial categorization and reasons as well as the frequently purchased-together products. Your answer should be chosen from the list:\n\n{}\n\n{}\n\nNow, apply this approach to the following target product.\n\n"
      #   prompts['product']= "Please predict the most likely category of the target product from Amazon based on its name and description as well as the frequently purchased-together products. Your answer should be chosen from the list:\n\n{}\n\n{}\n\nNow, apply this approach to the following target product.\n\n"
      #   prompts['product']= "Please predict the most likely category of the target product from Amazon based on its name, description, initial categorization and reasons as well as the frequently purchased-together products.\n\n{}\n\nNow, apply this approach to the following target product.\n\n"
    else:
        prompts['cora']= "Please predict the most appropriate category for the paper. Choose from the following categories:\n\n{}"
    
    prompts['cora_year']="Predict the 2 most probable publication time periods for the paper based on the titles and abstracts of the paper itself as well as its references and citations. Choose from the given list of time periods:\n\n{}\n\n{}"

    prompts['wisconsin']="Please predict the 2 most appropriate categories for the webpage based on the URL and category of the webpages which the target page has outbound links to or has inbound links from (for the non-main-page or resource pages labeled as 'other' within these webpages, their content abstract will be attached in addition) as well as the link pattern (inbound and outbound hyperlinks) of the target webpage. Choose from the following categories:\n\n{}\n\n{}"
    # prompts['wisconsin']="Please predict the 2 most appropriate categories for the webpage based on the URL and words with top 100 tf-idf of the target page and the webpages which the target page has outbound links to or has inbound linked from (for the non-main-page or resource pages labeled as 'other' within these webpages, their content abstract will be attached in addition) as well as the link pattern (inbound and outbound hyperlinks) of the target webpage. Choose from the following categories:\n\n{}\n\n{}"
    prompts['wisconsin']="Please predict the 2 most appropriate categories for the webpage based on the URL and content abstract of the webpage as well as its linked pages which the target page has outbound links to or has inbound links from. Choose from the following categories:\n\n{}\n\n{}"
    
    """\n\nHere is a brief explanation of each categories:

1. **Case Based**: Refers to case based reasoning, a method in artificial intelligence where new problems are solved by adapting solutions from similar past problems.

2. **Genetic Algorithms**: Represents papers on genetic algorithms, a class of optimization algorithms inspired by the process of natural selection in biology.

3. **Neural Networks**: Covers papers on neural networks, a set of algorithms modeled after the human brain, widely used in machine learning for tasks like image and speech recognition.

4. **Probabilistic Methods**: Encompasses research on probabilistic approaches in machine learning, including Bayesian networks and other methods that involve probability theory.

5. **Reinforcement Learning**: Includes papers on reinforcement learning, a type of machine learning where agents learn to make decisions by receiving rewards or penalties.

6. **Rule Learning**: Refers to methods focused on learning interpretable rules from data, often used in fields like data mining and knowledge discovery.

7. **Theory**: Represents theoretical research in machine learning, including the development of new algorithms and the mathematical foundations of machine learning."""
    refining2="""To categorize a paper based on the titles and abstracts of the paper itself as well as its references and citations, follow these steps:

1. **Read and Analyze the Main Paper**: Focus on the title and abstract to understand the central themes, methodologies, and research questions addressed by the paper.

2. **Examine References and Citations**:
   - **References**: Look at the titles and abstracts of the papers it references to get a sense of the foundational work and theoretical framework it builds upon.
   - **Citations**: Review the titles and abstracts of papers that cite this paper (if available) to gauge its impact and the areas where it has been considered relevant.

3. **Identify Key Terms**: Extract key terms and phrases from the main paper, its references, and citations that are indicative of specific research areas or disciplines.

4. **Match Terms with Categories**: Compare these key terms to the language typically associated with the provided categories. Look for overlap or recurring themes that align with one of those categories.

5. **Choose the Most Aligned Category**: Select the category that best matches the central themes and key terms identified in the analysis of the paper and its associated scholarly works.

This approach leverages the content of the paper and its scholarly network to make an informed decision about its categorization without additional tools or expert advice."""
    # prompts['cora']="Please further refine the categorization of the paper based on initial judgements, titles, abstracts, references, citations, and the reasons of the initial judgements. Choose from the following categories:\n\n{}\n\n{}"
    # Fetch the appropriate prompt
    prompt = prompts[source]
    #rec in cot rec5,7
    # refining = """To refine the initial categorization for the paper based on the initial categorizations and reasons regarding its title, abstract, references, and citations, you can follow a structured approach that leverages the interconnected information. Here's a detailed method:

    # 1. **Analysis of Initial categorization**:
    #     - **Consistency Check**: Compare the initial categories and reasoning of the main paper with those of its references and citations. Look for inconsistencies or patterns. For example, if many references are classified under a particular category that differs from the main paper, reconsider if the main paper might be more aligned with this category.
    #     - **Relevance Assessment**: Assess how directly the references and citations relate to the main content and focus of the paper.  Some references might be peripheral and have less impact on the category judgment than central, thematic references.
    # 2. **Contextual Integration**:
    #     - **Theme Synthesis**: Identify common themes and topics across the paper and its references/citations. This holistic view can provide insights into the most fitting category, especially if some themes are more prominent or recent in the research field.
    #     - **Trend Identification**: Consider if the paper aligns with any emerging trends evident in its citations. Papers often cite recent work that reflects the latest research directions, which might suggest a more suitable category.
    # 3. **Category Revision Proposal**:
    #     - **Proposed Adjustments**: Based on the above analysis, propose any adjustments to the paper's category. Clearly state why these changes are suggested, referencing specific details from the analysis (e.g., dominant themes in citations, emerging trends, inconsistency in initial categorization).
    #     - **Reasoning Documentation**: Document the reasoning for any proposed category revision comprehensively. This should include insights from the paper's content as well as influence from its references and citations.
    # 4. **Iterative Review**:
    #     - **Feedback Loop**: If possible, iteratively review the category adjustments. Reassess the proposed category in light of any new insights or overlooked details from the initial round of revision.
    #     - **Final Decision**: Confirm the final category assignment after thorough consideration, ensuring it best reflects the content and context of the paper and its academic network.

    # This method ensures that the refined categorization are not only based on the content of the papers themselves but also deeply informed by the academic discourse they engage with through references and citations. This approach leverages the interconnected nature of scholarly work to enhance the accuracy and relevance of category assignments."""
    refining = """To refine the initial categorization for the paper based on the initial categorizations and reasons regarding its title, abstract, references, and citations, you can follow a structured approach that leverages the interconnected information. Here's a detailed method:

    1. **Analysis of Initial categorization**:
        - **Consistency Check**: Compare the initial categories and reasoning of the main paper with those of its references and citations. Look for inconsistencies or patterns. For example, if many references are classified under a particular category that differs from the main paper, reconsider if the main paper might be more aligned with this category.
        - **Relevance Assessment**: Assess how directly the references and citations relate to the main content and focus of the paper.  Some references might be peripheral and have less impact on the category judgment than central, thematic references.
    2. **Contextual Integration**:
        - **Theme Synthesis**: Identify common themes and topics across the paper and its references/citations. This holistic view can provide insights into the most fitting category, especially if some themes are more prominent or recent in the research field.
        - **Trend Identification**: Consider if the paper aligns with any emerging trends evident in its citations. Papers often cite recent work that reflects the latest research directions, which might suggest a more suitable category.
    3. **Category Revision Proposal**:
        - **Proposed Adjustments**: Based on the above analysis, propose any adjustments to the paper's category. Clearly state why these changes are suggested, referencing specific details from the analysis (e.g., dominant themes in citations, emerging trends, inconsistency in initial categorization).
        - **Reasoning Documentation**: Document the reasoning for any proposed category revision comprehensively. This should include insights from the paper's content as well as influence from its references and citations.
    4. **Iterative Review**:
        - **Feedback Loop**: If possible, iteratively review the category adjustments. Reassess the proposed category in light of any new insights or overlooked details from the initial round of revision.
        - **Final Decision**: Confirm the final category assignment after thorough consideration, ensuring it best reflects the content and context of the paper and its academic network.
    5. **Final Check**: After final decision, review the reasoning process and final categorization carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.

    This method ensures that the refined categorization are not only based on the content of the papers themselves but also deeply informed by the academic discourse they engage with through references and citations. This approach leverages the interconnected nature of scholarly work to enhance the accuracy and relevance of category assignments."""

    refining4="""To iteratively refine the categorization of the papers when considering the paper itself and its references and citations, and where the correct category may be missed in previous iterations, you can adopt a more detailed and focused analysis. Here's how you can structure this iterative refinement process:

1. **Reevaluate Information**:
   - **Deep Dive into Content**: Re-read the abstracts, titles, and available reasoning of the paper, its references, and citations to uncover overlooked aspects or subtleties that could suggest a different category.
   - **Focus on Keywords and Themes**: Look for specific keywords, technical terms, and themes that are prevalent but may have been previously undervalued in the categorization process.

2. **Cross-reference Categories**:
   - **Comparison with Known Categories**: Compare the themes and topics you identify in the paper and its references/citations against the definitions or typical contents of the categories not previously considered.
   - **Category Definitions**: Revisit the definitions of the categories. There might be a need to reassess how well the paper aligns with a category that wasn't initially considered but appears relevant upon closer inspection.

3. **Analytical Redirection**:
   - **Hypothesis Testing**: Formulate hypotheses about possible correct categories based on the new insights and test these by comparing with the content of the references and citations.
   - **Look for Patterns**: Identify any recurring themes or methods in the paper and its network that align more closely with a different category than previously judged.

4. **Documentation of Findings**:
   - **Detailed Reasoning**: Document every new insight and the reasoning for leaning towards a different category. This should include detailed examples from the text of the paper, references, and citations.
   - **Comparison Notes**: Maintain notes on why other categories were ruled out in this iteration, based on the deepened analysis.

5. **Decision Making**:
   - **Propose Category Revisions**: Based on the comprehensive review, propose moving the paper to a more fitting category. Ensure that the reasons for this shift are well-documented and robust.
   - **Iterative Review**: Introduce a process for periodic review, where the categorization can be revisited to consider new understanding or approaches in categorization.

6.**Mock Peer Review**: Simulate a peer review by having hypothetical "external reviewers" provide feedback on the categorization based on the documentation and reasoning provided.

7. **Final Adjustment**:
   - **Adjust and Finalize**: Make the final adjustments to the category based on the accumulated insights and feedback.
   - **Update Records**: Ensure that all changes are well-recorded and that the dataset reflects these final categorizations accurately.

8. **Final Check**: After final decision, review the reasoning process and final categorization carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.

This iterative process focuses on uncovering and correcting oversights in initial categorization efforts, using a detailed and methodical approach to align the paper more accurately with the most appropriate category based on its content and academic context."""
    refining4="""To further refine the categorization for the paper, you MUST follow the following process. This process leverages detailed analysis, consistency checks, and thematic alignment while ensuring iterative refinement.

### Iterative Refinement Process ###

## 1. Reevaluate Information ##
- **Deep Dive into Content**: Re-read abstracts, titles, categories refined in last iteration and reasons for the paper, references, and citations to uncover overlooked aspects.
- **Focus on Keywords and Themes**: Identify specific keywords, technical terms, and prevalent themes that may have been undervalued.

## 2. Cross-reference Categories ##
- **Comparison with Known Categories**: Compare identified themes and topics against definitions or typical contents of categories not previously considered.
- **Category Definitions**: Revisit and clarify the definitions of the categories to ensure accurate alignment.

## 3. Analytical Redirection ##
- **Hypothesis Testing**: Formulate hypotheses about possible correct categories based on new insights and test them by comparing the content of references and citations.
- **Pattern Identification**: Identify recurring themes or methods in the paper and its network that align with a different category.

## 4. Define Refinement Criteria ##
- **Category Connotation Clarification**: Clearly define and document the connotation and boundaries of each category. All the categories in the given option list should be considered.

## 5. Conduct Refinement ##
- **Re-evaluate Initial Judgments**: Reassess the initial category judgments using clarified connotations and metrics.
- **Re-assess References and Citations**: Review the category assignments of references and citations, looking for emerging patterns.
- **Cross-check with Category Connotation**: Ensure assigned categories align with the refined understanding of each category.

## 6. Feedback Mechanism ##
- **Mock Peer Review**: Simulate a peer review by re-evaluating with a fresh perspective.
- **Iterative Review**: Introduce a process for periodic review to consider new understandings or approaches in categorization.

## 7. Final Adjustment and Check ##
Make the final adjustments to the category based on the accumulated insights and feedback. After final adjustment, review the reasoning process and final categorization carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.

By combining detailed analysis, consistency checks, and thematic alignment with iterative refinement, you can enhance the accuracy and relevance of category assignments for each paper. This comprehensive approach ensures that categorization is well-informed by both the content of the paper and its academic context.

### End of The Refinement Process ###"""

    refining4 = """To further revise the categorization for the paper using the available information (abstract, title, initial revision result, reasons for initial revision, and information about references and citations), you can follow these steps:

1. **Consolidate Information**:
   - **Main Paper**: Gather the title, abstract, initial category, revised category, and reasons for both initial and revised categorization.
   - **References and Citations**: Collect the same information for each reference and citation of the main paper.

2. **Analyze Discrepancies**:
   - **Compare Categories**: Compare the categories assigned to the main paper with those of its references and citations. Look for discrepancies or misalignments.
   - **Identify Overlooked Categories**: Identify any categories that were overlooked in both initial and revised categorization by analyzing the themes and topics present in the references and citations.

3. **Contextual Reassessment**:
   - **Thematic Clustering**: Group references and citations by thematic similarity. Identify dominant themes that might have been missed previously.
   - **Emerging Trends**: Check if any recent trends or emerging topics in the references and citations suggest a different category for the main paper.

4. **Detailed Content Analysis**:
   - **Deep Dive into Abstracts and Titles**: Re-examine the abstracts and titles of the main paper, references, and citations. Look for specific keywords, concepts, or phrases that might indicate a more appropriate category.
   - **Reevaluate Reasoning**: Reevaluate the reasons for the initial and revised categorization. Determine if any reasoning was based on superficial connections rather than substantive thematic alignment.

5. **Consider Influential References/Citations**:
   - **Influential Papers**: Identify which references and citations have the most significant influence on the main paper's content. These might be papers that are frequently cited or central to the main paper's arguments.
   - **Alignment with Influential Papers**: Assess if the main paper's content aligns more closely with the categories of these influential references and citations.

6. **Propose Revised Categories**:
   - **Category Adjustment**: Based on the detailed analysis, propose any necessary adjustments to the main paper's category. Consider if a category that was previously overlooked now seems more appropriate.
   - **Justification**: Provide clear justification for the proposed revision, referencing specific details from the paper's content, references, and citations.

7. **Iterative Refinement**:
   - **Feedback Loop**: Reassess the proposed category adjustments iteratively. Check for consistency and coherence in the categorization across the main paper and its references and citations.
   - **Final Decision**: Finalize the category decision after thorough review and consideration of all available information.

8. ****Final Cheeck**: After final decision, review the reasoning process and final categorization carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.

By following this structured approach, you can systematically identify and rectify any overlooked categories, ensuring that the categorization of each paper is as accurate and relevant as possible based on the available information."""

    refining_product="""Here's a step-by-step guide to categorize the target product from Amazon based on its name, description, and frequently purchased-together items:

---

### **Step 1: Analyze the Target Product**
1. **Review the Product Name:**
   - Extract key terms that indicate the product’s type, brand, or primary use.
   - Look for specific identifiers (e.g., "wireless headphones," "non-stick pan").

2. **Examine the Product Description:**
   - Identify primary features, materials, functionality, and target audience.
   - Note any keywords related to usage scenarios or specific industries.

---

### **Step 2: Investigate Frequently Purchased-Together Items**
1. **Identify Product Relationships:**
   - List the products commonly purchased with the target item.
   - Note the categories these items belong to (e.g., accessories, complementary items).

2. **Assess Contextual Use:**
   - Determine how these items relate to the target product (e.g., used together, replacement parts, or upgrades).
   - Look for common themes in functionality or audience needs.

---

### **Step 3: Match with Amazon Categories**
1. **Search for Keywords in Amazon’s Existing Categories:**
   - Refer to Amazon’s standard category hierarchy and locate keywords that align with the product name, description, and related items.

2. **Identify the Most Specific Fit:**
   - Choose a category that best matches the product’s primary function and the context provided by related items.
   - Prioritize categories with subcategories for precise placement.

---

### **Step 4: Validate the Category**
1. **Compare with Similar Products:**
   - Look up similar items on Amazon and note their assigned categories.
   - Ensure the target product aligns with these benchmarks.

2. **Consider User Intent:**
   - Reflect on how a typical customer would search for this product and validate the chosen category based on expected search behavior.

---

### **Step 5: Assign the Category**
- Use the collected data to confidently assign the most relevant category and subcategory for the product.

--- 

This approach ensures accuracy and consistency while leveraging all provided information."""
    refining_product5="""**System Prompt:**
As an expert product categorizer, your task is to select the single most appropriate Amazon product category for a target product. This dataset is an **interconnected co-purchase graph**. You must balance the product's fundamental physical/functional attributes with the topological context provided by "Frequently purchased-together items".

Please strictly follow this **Waterfall Decision Process**:

---

## Step 1: Core Function & Noise Filtration
- **Identify Core Purpose:** Determine what the item actually *does*. Do NOT be misled by secondary "smart" features.
- **Avoid Data Artifacts (CRITICAL):** Absolutely DO NOT select `label 25`, `#508510`, `Purchase Circles`, or `Gift Cards`. If the ground truth happens to be one of these, it is a dataset error. You must still predict the true logical category.
- **Missing Description:** If the product description is missing or vague, rely 100% on the co-purchased items to determine the category.

---

## Step 2: The Hub-Node Immunity (Absolute Priority 1)
- **Rule:** Universal electronic consumables (e.g., SD cards, AA/AAA batteries, generic USB drives) are IMMUNE to the graph ecosystem. An SD card is ALWAYS `Electronics` or `Computers`.

---

## Step 3: Hard-Boundary Vetoes & Specialized Domains (Absolute Priority 2)
Amazon taxonomy overrides graph topology based on physical form. Apply these vetoes IMMEDIATELY:
- **The OPE Veto (Outdoor Power Equipment):** Chainsaws, lawnmowers, and their specific accessories/protective apparel (e.g., Husqvarna chaps/helmets) MUST go to `Patio, Lawn & Garden` (NOT Tools or Clothing).
- **The Installation & Fixture Veto:** Hardwired items (wall plates, built-in fixtures, permanent lighting) MUST go to `Tools & Home Improvement`. *(Exception: TV wall mounts/brackets go to `Electronics` per the Host-Device rule).*
- **The Painting & Color Veto:** Pantone color guides and paint swatches are painting tools and MUST go to `Tools & Home Improvement`.
- **The Appliance Veto:** Humidifiers, air purifiers, and heaters (even if marketed specifically for pets or terrariums) MUST go to `Home & Kitchen` (or Appliances).
- **The Apparel & Luggage Veto:** Everyday wearables AND everyday luggage (backpacks, suitcases) MUST be `Clothing, Shoes & Jewelry`. *(Exceptions: Chainsaw chaps -> Patio; Party masks -> Toys).*
- **The Toys, Party & Kids' Crafts Veto:** Action figures, Anime PVC statues, party supplies, festive decorations, AND Kids' Arts & Crafts (Crayola, coloring books) MUST go to `Toys & Games`. (Leave `Collectibles & Fine Art` ONLY for rare coins and signed memorabilia).
- **The Educational Veto:** Classroom decorations and teacher supplies MUST go to `Office Products`.
- **The Furniture & Wall Art Veto:** Posters, prints, wall art, or large storage furniture MUST default to `Home & Kitchen`. Ignore small co-purchased pens/tools.
- **The Pro-Audio Veto:** Professional studio/stage audio gear (standalone XLR microphones, mixers) MUST go to `Musical Instruments`. *(Exception: Audio adapters, DI boxes, and generic A/V cables stay in `Electronics`).*
- **The Optics & Photography Veto:** Telescopes and binoculars belong to `Camera & Photo`. However, Studio Lighting Kits, Strobes, and Softboxes map to `Electronics`.
- **The Physical Media Veto:** Physical books -> `Books`. Music albums/bands/tracks -> `CDs & Vinyl`. Workout DVDs/Films -> `Movies & TV`. Digital books -> `Kindle Store`.
- **The Edible & Supplement Veto:** Whole foods or natural snacks MUST be `Grocery & Gourmet Food`. *(Exception: Superfoods like cacao nibs co-purchased with vitamins MUST go to `Health & Personal Care`).*

---

## Step 4: The Ultimate Subordination Rule & Graph Inference (The Default Engine)
In Amazon's taxonomy, the **Target of Service (Host)** ALWAYS overrides the physical environment or form:
- **If the Host is a Device (Phones/TVs/Cameras):** The accessory inherits the device's category. (e.g., Cell phone car mount -> `Cell Phones & Accessories`; TV Wall Mount -> `Electronics`).
- **If the Host is a Vehicle (Cars/Motorcycles):** The tool, part, or fluid inherits the vehicle's category. (e.g., Car wax -> `Automotive`).
- **Absolute Graph Submission for Generic Materials:** Generic raw materials (e.g., elastic cord) are 100% dictated by their graph neighbors.
- **Domain over Form:** Generic functional items inherit graph neighbors. Tape and adhesives (even "craft tape") usually default to `Tools & Home Improvement`.
- **Ignore Cart-Fillers:** Disregard obvious random co-purchases like snacks/cookies.

---

## Step 5: Resolve Amazon-Specific Traps & Decoys
1. **Household Supplies:** Laundry detergents, dish soaps, AND disposable tableware (plastic cups, paper plates, toilet paper) MUST go to `Health & Personal Care` (NOT Home & Kitchen).
2. **Skin & Lip Care:** Sunscreens and facial cleansers go to `Beauty`. However, medicated Lip Balms/Protectants go to `Health & Personal Care`.
3. **Specialized Outdoor First-Aid:** General first-aid is Health, but Backpacking/Camping-specific survival kits go to `Sports & Outdoors`.
4. **Casino & Game Room:** Casino equipment and card shufflers belong to `Sports & Outdoors`.
5. **Industrial & 3D Printing:** 3D printer filaments and heavy commercial supplies go to `Industrial & Scientific`.
6. **Cable Management & PC Components:** Cord protectors, wire raceways, internal Hard Drives (HDD/SSD), laptop screens, and bare internal components map to `Electronics` (NOT Computers).
7. **Coolants vs. First Aid:** Generic ice packs for food belong to `Home & Kitchen`.
8. **Bath & Body:** Artisanal soaps, body washes, and bath sponges -> `Beauty`.
9. **Decoy Label Avoidance (CRITICAL):** - ALWAYS prefer `Home & Kitchen` for cookware/bakeware. **NEVER use `Kitchen & Dining` or `Furniture & Decor`.**
   - ALWAYS prefer `Office Products`. **NEVER use `Office & School Supplies`.**
   - ALWAYS prefer `Baby Products`. **NEVER use `Baby`.**

---

## Step 6: Master Sanity Checklist (MENTAL VERIFICATION)
Before outputting, mentally verify against these structured trap-checks:

**[ ] A. Lifestyle & Fashion Traps:**
- Backpack/Luggage? -> FORCE `Clothing, Shoes & Jewelry`.
- Kids' Craft/Crayola/Party supply? -> FORCE `Toys & Games`.
- Casino/Poker equipment? -> FORCE `Sports & Outdoors`.

**[ ] B. Home & Hardware Traps:**
- Cookware/Bakeware/Ice Packs? -> FORCE `Home & Kitchen` (Avoid Kitchen & Dining).
- Hardwired fixture or Pantone guide? -> FORCE `Tools & Home Improvement`.
- Humidifier/Heater? -> FORCE `Home & Kitchen`.

**[ ] C. Tech & Media Traps:**
- TV Mount / Studio Light Kit / Audio Cable / HDD? -> FORCE `Electronics`.
- Pro XLR Mic / Audio Mixer? -> FORCE `Musical Instruments`.
- Music Album/Book/DVD? -> Match specific Physical Media category.

**[ ] D. Health & Outdoors Traps:**
- Plastic Cups / Toilet Paper / Detergent? -> FORCE `Health & Personal Care`.
- Lip Balm (medicated) / Superfoods+Vitamins? -> FORCE `Health & Personal Care`.
- Chainsaw gear / Outdoor Power Equipment? -> FORCE `Patio, Lawn & Garden`.

**[ ] E. Graph & Decoy Traps:**
- Generic material (e.g., elastic cord)? -> MUST follow graph neighbors.
- Did I pick a Decoy (`Office & School Supplies`, `Kitchen & Dining`, `#508510`)? -> CHANGE IT.
"""
#     refining="""To further revise the categorization of the paper, considering the possibility of overlooked correct categories in the initial categorization and revision, you can employ a more systematic and iterative approach using the same available information. Here's a detailed process:

# ### 1. Gather and Organize Information
# - **Title and Abstract**: The primary source of understanding the paper's content.
# - **Initial Categorization**: The categories assigned initially along with the reasoning.
# - **Initial Revision**: Adjusted categories and the reasons for those adjustments.
# - **References and Citations**: Categories and reasoning for each reference and citation.

# ### 2. Re-evaluate Categories Based on Content
# - **Keyword Analysis**: Extract key terms and phrases from the title and abstract that define the core topics and contributions of the paper. Compare these with the key terms from its references and citations.
# - **Thematic Mapping**: Create a thematic map that connects the paper with its references and citations based on common keywords and topics. This visual representation can highlight overlooked areas or themes.

# ### 3. Reassess Consistency and Relevance
# - **In-depth Comparison**: Examine the consistency of the paper's categorization with its references and citations:
#   - **References**: If a majority of references fall under a specific category that was not considered or given less weight initially, reassess the main paper's category.
#   - **Citations**: Recent citations often indicate the paper's impact and evolving relevance. Consider if these suggest a different or additional category.
# - **Relevance Weighting**: Assess the relevance of each reference and citation to the core content of the paper. Higher relevance should have more influence on the category decision.

# ### 4. Identify and Correct Overlooked Categories
# - **Broader Perspective**: Revisit the title and abstract to ensure no key aspect was missed. Sometimes initial revisions might focus too narrowly on specific terms.
# - **Alternative Categorization**: Consider alternative perspectives for categorization based on broader or interdisciplinary connections observed in the references and citations.

# ### 5. Refine Reasoning and Justifications
# - **Detailed Justifications**: For any category adjustments, provide detailed reasoning. Include how the abstract, title, and themes from references and citations support the new categorization.
# - **Cross-check Overlaps**: Verify if the new categories overlap with initial and revised judgments. Ensure that the reasons for overlaps are well documented and justified.

# ### 6. Iterative Refinement Process
# - **Review Loop**: Perform a review loop where you:
#   - Reassess the categories considering the new findings.
#   - Document the reasoning for any changes.
#   - Compare the revised categorization with initial judgments to ensure consistency and accuracy.
# - **Consistency Check**: Ensure that the new category aligns with the broader context of the paper's references and citations, ensuring no significant thematic areas are missed.

# ### 7. Final Documentation and Update
# - **Clear Documentation**: Keep clear and detailed records of the final category, the reasons for the final judgment, and how it differs from initial categorizations.
# - **Final Cheeck**: After final judgement, review the reasoning process and final categorization carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.

# By following this structured approach, you can ensure a thorough and accurate categorization process, leveraging all available information and iteratively refining your judgments."""

    #rec in cot rec4
    # refining = "To further refine the initial categorization for the paper based on initial categorizations, titles, abstracts, references, citations and the reasons of the initial categorizations, you can follow a systematic approach. Here's a step-by-step process that you can implement:\n\n"+\
    # "### 1. **Review Initial Categorizations and Rationales**\n"+\
    # "   - Begin by thoroughly examining the initial category assignments and the reasons provided for these choices. Understand why certain categories were deemed appropriate based on the content of the paper, its abstract, title, and the nature of its references and citations.\n\n"+\
    # "### 2. **Cross-reference with Cited Papers**\n"+\
    # "   - Look into the categories of the papers cited by the main paper. Papers often cite works that are within the same or closely related fields. If a majority of the references belong to a particular category, this might reinforce or question the initial judgement of the main paper's category.\n"+\
    # "   - Similarly, consider how the main paper is cited by other works. This can provide insight into how the scholarly community perceives the relevance of the paper to certain categories.\n\n"+\
    # "### 3. **Analyze Keywords and Themes**\n"+\
    # "   - Extract and analyze the main keywords and themes from the title and abstract of the paper. Check if these keywords align more closely with one category over another.\n"+\
    # "   - Identify the frequency and relevance of specific terms that are closely associated with the categories in the given list.\n\n"+\
    # "### 4. **Feedback Loop**\n"+\
    # "   - Iterate the categorization process based on the insights gained from the above steps. Each cycle should refine the understanding and placement of the paper within the most fitting category.\n"+\
    # "   - It's also beneficial to look for any conflicting signals or anomalies (e.g., a paper heavily citing literature from a different category than initially judged) and resolve these by deeper analysis of the content and context.\n\n"+\
    # "### 5. **Consistency Check**\n"+\
    # "   - Ensure that the revised categorization is consistent with the categorizations of similar papers. This can be checked manually or by comparing the features (keywords, referenced works, etc.) of the paper with those in the same category.\n\n"+\
    # "### 6. **Final Validation**\n"+\
    # "   - After adjustments, review the paper's categorization to ensure it logically fits with its content and its scholarly context. This final check ensures that the category reflects the paper's contributions and themes accurately.\n\n"

    refining_cora_year="""Predicting the publication time period of a paper based on titles and abstracts of its references and citations, without knowing their exact publication dates, is a challenging task. However, it is possible to make educated guesses based on the evolution of topics, terminology, and methodologies over time, especially in a fast-evolving field like machine learning. Here's how you can approach this task:

1. **Understand the Timeline of Machine Learning**: Familiarize yourself with the historical development of machine learning, noting when key concepts, algorithms, and techniques were introduced and became popular. This will help you recognize which era the content of a paper most likely aligns with.

2. **Analyze References**:
    - **Topic Trends**: Identify the topics covered in the references. Certain topics or methods can indicate a specific time period. For example, deep learning has been particularly prominent since the mid-2010s.
    - **Terminology**: Pay attention to specific terms used. Terminology evolves, and some terms may either be outdated or recent.
    - **Methodological Advances**: Note any mention of methodologies that are tied to particular breakthroughs or advancements. For example, references to convolutional neural networks (CNNs) might suggest a timeframe post-2010.

3. **Evaluate Citations**:
    - **Current Relevance**: Citations can provide insights into the paper's relevance in contemporary research. A paper frequently cited by more recent studies might suggest a closer publication date to those studies.
    - **Technological Context**: Consider the technology or datasets mentioned in citations. For example, references to specific GPU architectures or large datasets like ImageNet could hint at more recent work.

4. **Synthesize Insights**:
    - **Cross-reference Findings**: Combine observations from both references and citations to align with known timelines in machine learning. For instance, if references largely discuss neural networks in a way that predates deep learning, the paper might belong to an earlier period.
    - **Majority Context**: If the majority of thematic and methodological indicators point to a specific era, that might be the most probable publication time period.

5. **Decision Making**:
    - **Estimate a Range**: Based on the predominant time period indicators from both references and citations, estimate a probable range of years or decades.
    - **Use a Cautious Approach**: If the evidence is mixed or unclear, consider a broader time period that encompasses all plausible options.

6. **Document Your Process and Reasoning**: Keeping a record of how you arrived at each estimation can help in refining the process for future predictions and providing justification for your choices.

7. **Final Review**: 
    - **Self-Check**: Review the analysis process and the final prediction carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure that the analysis process is logically correct and fitting with the paper itself as well as its citation network, the final prediction aligns with the overall analysis and that the paper is placed where it best fits within the academic landscape.

This method leverages the fact that academic papers generally build upon prior work and are subsequently built upon by newer work. By analyzing how the paper fits within the known evolution of machine learning, you can make a reasonable guess about when it was likely published.
"""
    refining_wisconsin="""Classifying these pages involves understanding the structure and patterns of hyperlink relationships and using the content abstracts of "other" pages. Here's a step-by-step approach to how you can manually classify these pages:

### Step-by-Step Classification Process

1. **Identify Link Patterns**:
   - Examine the outgoing and Inbound Links for each target page.
   - Note the categories of pages each target page links to and is linked from.

2. **Categorize Based on Link Patterns**:
   - **Faculty Pages**:
     - Typically, a faculty page will have incoming links from the department page, course pages, student pages, other faculty pages and project pages.
     - It will likely link out to course pages, department pages, other "other" pages such as publications, research interests, and vitae.
   - **Student Pages**:
     - Often linked from the course pages, students directories, project pages.
     - May link out to course pages, faculty pages, department pages, personal project pages and possibly some "other" resource pages like their research work, personal bio or non-acadmic personal content.
   - **Course Pages**:
     - Will likely have incoming links from course directories, faculty pages, student pages and other course pages.
     - They usually link to course resource pages, various student and project pages, as well as faculty pages.
   - **Department Pages**:
     - Typically have the highest level of inbound links from various categories (faculty, students, courses, staff, projects).
     - They will link out to all other main categories.
   - **Staff Pages**:
     - Likely linked from department and project pages.
     - May link to various "other" resource pages, indicating administrative functions.
   - **Project Pages**:
     - Have links from project directories, faculty and student pages.
     - They link to other related project pages, faculty and student pages and occasionally department pages.

3. **Utilize Content Abstracts for "Other" Pages**:
   - For pages classified as "other", read their content abstracts.
   - Determine if the content aligns more closely with research publications (faculty), coursework materials (course), personal student research (student), personal bio (student), administrative content (staff), or project descriptions (project).

4. **Cross-reference Relationships**:
   - Validate your preliminary classifications by cross-referencing hyperlink relationships.
   - For example, if a page is preliminarily classified as a faculty page but lacks inbound links from department pages, reassess based on content abstracts or additional hyperlink patterns.

5. **Iterative Review**:
   - Before you categorize the target webpage as "other", carefully determine which category it aligns more closely with, for examples, research publications (faculty), coursework materials (course), personal student research (student), personal bio (student), administrative content (staff), or project descriptions (project). Don't rush to categorize it as "other" unless it's impossible a main page of course, project, student, faculty, staff or department.
   - Continuously review and refine the classification as more patterns emerge.
   - Ensure consistency across similar types of pages.
   - Review the analysis process and the final category carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure that the analysis process and final category are logically correct and fitting with the its context.

By carefully analyzing the hyperlink patterns and using the content abstracts for "other" pages, you can manually classify the webpages into the appropriate categories.
"""
    refining_wisconsin="""To classify these pages manually without the aid of a machine learning model, you can follow a systematic approach by analyzing the available data (link patterns and content abstracts for "other" pages). Here's a step-by-step guide to help you with the classification process:

### Step 1: Analyze Link Patterns
Link patterns can provide significant clues about the category of a webpage. Consider both the inbound and outbound links:

1. **Inbound Links:**
   - **From Faculty Pages:** Indicates the target page may be important for faculty members, possibly related to faculty, projects, or research. Somtimes students also have single inbound link from faculty.
   - **From Student Pages:** Indicates relevance to students, potentially a course, faculty, department, or project page.
   - **From Course Pages:** Likely indicates a relationship with academic courses for example, faculty, student or course materials.
   - **From Department Pages:** May suggest the page is important for the entire department, possibly an administrative or project page.
   - **From Staff Pages:** Could imply relevance to staff-related activities.
   - **From Project Pages:** Suggests involvement in specific projects, may be faculty or students.
   - **From Other Pages:** Determine according to the content of this "other" page. For example, inbound links from student directories or departmental graduate student portal page indicate it's a student page, inbound links from course list indicate it's a course page, etc. 

2. **Outbound Links:**
   - **To Faculty Pages:** The target page could be related to faculty activities or information, for example, it may be a student page the faculty is supervising, the project page the faculty is leading, or the course page the faculty is teaching, etc.
   - **To Student Pages:** The target page might be course, faculty pages, or it providing resources or information useful/related to students such as students directories. 
   - **To Course Pages:** Likely to be faculty/student pages as teacher/TA, or it is related to academic content or course information.
   - **To Department Pages:** The personal mainpages (staff/faculty/student) has outbound links back to department mainpage. It indicates a broader departmental focus if it's an "other" page. 
   - **To Staff Pages:** Might be providing staff-related resources or information.
   - **To Project Pages:** Suggests the target page is project-related, for example, it's a faculty page or student page as a participant.
   - **To Other Pages:** Consider the content of the "other" page to infer the target page's category. For example, outbound links to research publications indicate a faculty/student page, outbound links to miscellaneous personal content indicate a student page, outbound links to administrative documents indicate a staff page, outbound to course materials indicate a course page, etc.

### Step 2: Content Abstract Analysis (for "Other" Pages)
For pages labeled as "other" with attached content abstracts, examine the abstracts closely to understand the primary focus of the page. Look for keywords and phrases that indicate:

- **Academic Terminology:** Course names, academic terms, syllabus details, etc., indicate course pages.
- **Research Terminology:** Research interests, publications, projects, etc., suggest faculty or project pages.
- **Administrative Language:** Departmental policies, staff roles, administrative announcements, etc., indicate department or staff pages.

### Step 3: Contextual Linking and Category Inference
Integrate the insights from link patterns and content abstracts to infer the category:

1. **Faculty Pages:**
   - High number of inbound links from course, project, students ot other faculty pages. Sometimes inbound links from research group/ laboratory related content.
   - Outbound links to research interests, publications, projects related content, courses related content or departmental resources. Sometimes to student pages.

2. **Student Pages:**
   - Inbound links from course, projec, student directories, departmental graduate student portal or other student pages. Sometimes a single inbound link from faculty page. Sometimes inbound links from groups (both acadmic or miscellaneous are possible). 
   - Outbound links to course-related content, project related content, faculty members or miscellaneous personal content. Sometimes with a single outbound link to department mainpage.

3. **Course Pages:**
   - Inbound links from student pages or faculty pages.
   - Outbound links to faculty members, students, syllabi, and academic resources.

4. **Department Pages:**
   - High number of inbound links from all categories (faculty, student, staff, etc.).
   - Outbound links to department-wide resources (including content related to research, students, faculty, course, administrative, etc.).

5. **Staff Pages:**
   - Inbound links from departmental or faculty pages.
   - Outbound links to administrative documents, departmental resources.

6. **Project Pages:**
   - Inbound links from faculty and student pages.
   - Outbound links to research-related content, publications, participants (faculty/student) and external resources.

7. **Other Pages:**
   - Typically have content abstracts attached.
   - Serve as auxiliary pages linked primarily to a main category page (faculty, student, etc.).

### Step 4: Cross-Verification
Verify your initial classification by cross-referencing with multiple link patterns and content abstracts. If a page seems ambiguous, check the link patterns again and reassess based on the most frequent category of linking pages.

### Step 5: **Final Review**: 
Review the analysis process and the final category carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure that the analysis process and the final category of target page are logically correct and fitting with its context.

By carefully analyzing the link patterns and content abstracts, you can systematically classify the webpages into their respective categories.
"""
    refining_wisconsin="""To classify these pages manually without the aid of a machine learning model, you can follow a systematic approach by analyzing the available data (link patterns and content abstracts for "other" pages). Here's a step-by-step guide to help you with the classification process:

### Step 1: Analyze Link Patterns
Link patterns can provide significant clues about the category of a webpage. Consider both the inbound and outbound links:

1. **Inbound Links:**
   - **From Faculty Pages:** Indicates the target page may be important for faculty members, possibly related to faculty, projects, or research. Somtimes students also have single inbound link from faculty.
   - **From Student Pages:** Indicates relevance to students, potentially a course, faculty, department, or project page.
   - **From Course Pages:** Likely indicates a relationship with academic courses for example, faculty, student or course materials.
   - **From Department Pages:** May suggest the page is important for the entire department, possibly an administrative or project page.
   - **From Staff Pages:** Could imply relevance to staff-related activities.
   - **From Project Pages:** Suggests involvement in specific projects, may be faculty or students.
   - **From Other Pages:** Determine according to the content of this "other" page. For example, inbound links from webpages that list information of graduate students (such as student directories) indicate it's a student page, inbound links from course list indicate it's a course page, etc. 

2. **Outbound Links:**
   - **To Faculty Pages:** The target page could be related to faculty activities or information, for example, it may be a student page the faculty is supervising, the project page the faculty is leading, or the course page the faculty is teaching, etc.
   - **To Student Pages:** The target page might be course, faculty pages, or it providing resources or information useful/related to students such as students directories. 
   - **To Course Pages:** Likely to be faculty/student pages as teacher/TA, or it is related to academic content or course information.
   - **To Department Pages:** The personal mainpages (staff/faculty/student) has outbound links back to department mainpage. It indicates a broader departmental focus if it's an "other" page. 
   - **To Staff Pages:** Might be providing staff-related resources or information.
   - **To Project Pages:** Suggests the target page is project-related, for example, it's a faculty page or student page as a participant.
   - **To Other Pages:** Consider the content of the "other" page to infer the target page's category. For example, outbound links to research publications indicate a faculty/student page, outbound links to miscellaneous personal content indicate a student page, outbound links to administrative documents indicate a staff page, outbound to course materials indicate a course page, etc.

### Step 2: Content Abstract Analysis (for "Other" Pages)
For pages labeled as "other" with attached content abstracts, examine the abstracts closely to understand the primary focus of the page. Look for keywords and phrases that indicate:

- **Academic Terminology:** Course names, academic terms, syllabus details, etc., indicate course pages.
- **Research Terminology:** Research interests, publications, projects, etc., suggest faculty or project pages.
- **Administrative Language:** Departmental policies, staff roles, administrative announcements, etc., indicate department or staff pages.

### Step 3: Contextual Linking and Category Inference
Integrate the insights from link patterns and content abstracts to infer the category:

1. **Faculty Pages:**
   - High number of inbound links from course, project, students or other faculty pages.
   - Outbound links to research interests, publications, projects related content, courses related content or departmental resources. Sometimes to student pages.

2. **Student Pages:**
   - Inbound links from course, project or other student pages. Often with an inbound link from webpages that list information of graduate students (such as student directories). Sometimes a single inbound link from faculty page.
   - Outbound links to course-related content, project related content, faculty members or miscellaneous personal content. Sometimes with a single outbound link to department mainpage.

3. **Course Pages:**
   - Inbound links from student pages or faculty pages.
   - Outbound links to faculty members, students, syllabi, and academic resources.

4. **Department Pages:**
   - High number of inbound links from all categories (faculty, student, staff, etc.).
   - Outbound links to department-wide resources (including content related to research, students, faculty, course, administrative, etc.).

5. **Staff Pages:**
   - Inbound links from departmental or faculty pages.
   - Outbound links to administrative documents, departmental resources.

6. **Project Pages:**
   - Inbound links from faculty and student pages.
   - Outbound links to research-related content, publications, participants (faculty/student) and external resources.

7. **Other Pages:**
   - Typically have content abstracts attached.
   - Serve as auxiliary pages linked primarily to a main category page (faculty, student, etc.).

### Step 4: Cross-Verification
Verify your initial classification by cross-referencing with multiple link patterns and content abstracts. If a page seems ambiguous, check the link patterns again and reassess based on the most frequent category of linking pages.

### Step 5: **Final Review**: 
Review the analysis process and the final category carefully and identify any factual errors, inconsistencies, or missing important information. If you find any issue, please fix it accordingly to ensure that the analysis process and the final category of target page are logically correct and fitting with its context.

By carefully analyzing the link patterns and content abstracts, you can systematically classify the webpages into their respective categories.
"""

    # v2（2026-06，针对新一代模型 TAPTN1 逐例诊断修订；见 PLAN E1-f）。
    # 原 refining_wisconsin 三处缺陷：(1) faculty 规则"有论文/Vita/研究"会吞并研究型 staff；
    # (2) 缺"研究组/实验室着陆页(URL 形如 /users/xxx 但实为 project)"判别，被误判为个人页；
    # (3) faculty/student/staff 在仅被研究组链接、无授课/指导信号时缺平局判别锚。
    # v2 只在原指令末尾追加这三条补充判别规则，不改动原 5 步流程；TAPTN_INSTR_V2=1 启用。
    refining_wisconsin_v2 = refining_wisconsin.rstrip() + """

### Additional Disambiguation Rules (apply during Step 3 and Step 5)

**A. Staff vs. Faculty (do NOT let research output alone imply faculty).**
A personal page that lists publications, a Vita/CV, or research interests is NOT necessarily faculty. Research scientists, systems/lab managers, and technical staff also publish and keep such pages. Classify as **staff** (not faculty) when the page shows ANY of: links to or responsibility for computer/lab *facilities*, *systems administration*, or *technical support*; an operations/service role; OR the absence of faculty-defining signals below. Classify as **faculty** only when there is a positive faculty signal: a title such as Professor/Associate Professor/Assistant Professor/Lecturer, evidence of *teaching a course*, or *advising/supervising students*. If a page has research output but none of these faculty-defining signals and shows any technical/operational duty, prefer **staff**.

**B. Research-group / lab landing page is a `project`, even under `/users/`.**
A page is a **project** page (rank it FIRST, above faculty/student) when it acts as a hub for a research group or lab: it links to MULTIPLE group members (several faculty and/or students), and/or it hosts the group's projects, publications, or software, and/or it is named after a group rather than one person — REGARDLESS of whether its URL looks personal (e.g., `/users/<name>`) or whether it is associated with a leading professor. Do not default such hub pages to the leading professor's "faculty" page; the page represents the group/project, not the individual.

**C. Personal-page tie-breaker (faculty vs. student vs. staff).**
When a personal page is linked only from research-group/lab hubs and the role is ambiguous, decide by positive signals, not by inbound-link count alone:
- **faculty**: holds a faculty title, teaches a course, or advises students;
- **student**: an inbound link from a graduate-student directory/listing, mention of an advisor/PhD program, or being listed among "graduate students";
- **staff**: maintains facilities/systems/technical-support resources, or holds a service/operations role with no teaching and no advising.
Only fall back to the generic "personal page → faculty/student" heuristic when none of the above positive signals are present.\n\n"""

    # v3（2026-06-13，对 v2 的逐例回归再迭代；见 PLAN E1-h）。v2 的规则 B/C 过校正：
    # (1) 规则B把"教授本人主页(仅链接到自己实验室)"误升为 project（node41 faculty→project 平局）；
    # (2) 规则C侵蚀了"无信号个人页默认 faculty"的先验，把无信号页推向 student/staff（node2 faculty→student；
    #     并因抬高 student 提及间接令抽取器误抓，node770 course→student）。
    # v3 收紧 B（要求"多成员/群体为主体"且对带 faculty 信号的页做豁免）、收紧 C（无正向信号时回落 faculty）、
    # 并加去平局约束（缓解 Texas 抽取在 1.0/1.0 时抓错）。TAPTN_INSTR_V2=1 启用（指向最新 v3）。
    refining_wisconsin_v3 = refining_wisconsin.rstrip() + """

### Additional Disambiguation Rules (apply during Step 3 and Step 5)

**A. Staff vs. Faculty (do NOT let research output alone imply faculty).**
A personal page that lists publications, a Vita/CV, or research interests is NOT necessarily faculty. Research scientists, systems/lab managers, and technical staff also publish and keep such pages. Classify as **staff** (not faculty) ONLY when the page shows a positive staff signal: links to or responsibility for computer/lab *facilities*, *systems administration*, *technical support*, or an operations/service role. Classify as **faculty** when there is a positive faculty signal: a title such as Professor/Associate/Assistant Professor/Lecturer, an inbound link from a faculty/people directory, evidence of *teaching a course*, or *advising/supervising students*. (Research output without any of these is NOT enough to decide either way — fall through to Rule C.)

**B. Research-group / lab landing page is a `project` — but only a genuine multi-member hub.**
Classify a page as **project** (and rank it FIRST) ONLY when its PRIMARY content is the research group/lab itself: it links to MULTIPLE distinct members (several faculty and/or students) and/or hosts the group's collective projects, publications, or software, and is centered on the group rather than on one named individual — REGARDLESS of a personal-looking URL (e.g., `/users/<name>`).
Important carve-out (do NOT over-apply B): if the page carries a positive *personal faculty* signal (a Professor/Lecturer title, or an inbound link from a faculty/people directory), classify it as **faculty**, even when it links out to "their" lab or group. A faculty member's own homepage that merely points to their lab is NOT a group hub and is NOT a project page.

**C. Personal-page tie-breaker (faculty vs. student vs. staff), with a faculty default.**
When a personal page is ambiguous (e.g., linked only from research-group/lab hubs, no abstract), decide by positive signals:
- **student**: an inbound link from a graduate-student directory/listing, or an explicit advisor/PhD-program mention;
- **staff**: maintains facilities/systems/technical-support resources, or holds a service/operations role with no teaching and no advising;
- **faculty**: a faculty title, teaching, or advising.
Default rule: if NONE of the above positive student/staff signals is present, a signal-less personal page (e.g., `/users/<name>`, `/profiles/<name>`) in a CS department defaults to **faculty** (the dominant class for such pages). Do NOT downgrade a signal-less personal page to student/staff merely because it is linked from a project/lab hub.

**D. Break ties.** Assign DISTINCT relevance scores to your top categories; do not output two categories with the same score. If two seem equal, use Rules A–C to pick one and give it a strictly higher score, so the single highest-relevance category is unambiguous.\n\n"""

    # 2hop（2026-06-15，2-hop/邻域聚合专用；见 PLAN E1-h2）。基底取最佳 1-hop 指令 v3，
    # 再追加两条"如何使用邻居初判"的规则，治理 2-hop 的唯一受害模式：邻居驱动的锚定漂移。
    # 跨模型 2-hop(v2) 检验：Gemma +0.78、GLM +0.39（只修复零受害），Qwen 唯一受害=node687
    # (gold=student，1-hop 已正确 0.95，被 project 邻居813 拉偏升为 project)。
    # 根因：(i) v2/v3 是 1-hop 指令，未规定邻居初判仅为弱先验、不得推翻强自证据的 round-1 主类；
    #       (ii) Rule B 的"研究组枢纽→project"未区分"枢纽页 vs 成员页"，令参与 project 的个人页被误升。
    # 规则 E/F 经 4 节点对照校验：保 Gemma{2,820}/GLM{770} 修复（其 1-hop 自证据弱/错，允许修订），
    #       治 Qwen{687} 受害（1-hop 自证据强且正确，禁止被邻居推翻；成员页≠project）。
    # 启用：WISCONSIN_VER=2hop。
    _two_hop_addendum = """

### Two-Hop Neighborhood-Aggregation Rules (apply ONLY when neighboring pages' preliminary categories are provided)

The neighboring pages' preliminary categories come from an independent earlier round. Treat them as NOISY contextual hints that may help break a genuine tie — never as ground truth, and never as an override of the target page's own evidence.

**E. Self-evidence priority (anti-drift).** If your own analysis of the target page assigns a confident, well-grounded PRIMARY category from the page's OWN type signals (a clear personal student/faculty page, a course syllabus, a department index, a genuine group hub), KEEP that as the primary category even when one or more neighbors belong to a different class. A neighbor's category may influence only your SECOND choice; it must never demote a confident, self-grounded primary category. Revise your primary category ONLY when its own evidence is weak, missing, or tied (near-equal top scores) AND the neighborhood consistently and plausibly points to a different, well-justified type.

**F. Member vs. hub (project disambiguation under aggregation).** Linking to, being linked from, or participating in a project/research-group/lab does NOT make an individual's page a `project`. The `project` class is reserved for the GROUP HUB page itself — the page whose SUBJECT is the group (it lists multiple distinct members and/or hosts the group's collective software, publications, or announcements). An individual who merely participates in or belongs to a group remains `student`/`faculty`/`staff` according to their own page type. Do NOT raise `project` above a confident student/faculty/staff primary merely because a neighbor or an outbound link is a project page. (Rule B still classifies a true multi-member hub as `project`; Rule F only blocks promoting member pages.)\n\n"""

    # 2hop = 最佳 1-hop 基底 v3 + 2-hop 邻域聚合规则 E/F（启用：WISCONSIN_VER=2hop）。
    refining_wisconsin_2hop = refining_wisconsin_v3.rstrip() + _two_hop_addendum
    # v2_2hop = v2 基底 + E/F，用于"纯 v2 的 2-hop 改进"消融（与 v2-2hop 同基底，仅加 E/F；
    # 隔离 E/F 效果，避免 v2→v3 基底切换混淆）。启用：WISCONSIN_VER=v2_2hop。
    refining_wisconsin_v2_2hop = refining_wisconsin_v2.rstrip() + _two_hop_addendum

#     refining_wisconsin="""To categorize a target main webpage using its content abstract, URL, and links to/from other pages, follow these systematic steps:

# ---

# ### **Step 1: Data Collection**
# 1. **Extract Content Features**:
#    - Analyze the **content abstract** of the target page.
#    - Identify keywords, phrases, and topics related to specific categories (e.g., faculty, student, course, etc.).

# 2. **Analyze URL Structure**:
#    - Examine the target webpage's **URL** for patterns or keywords that might hint at its category (e.g., `/faculty/`, `/course/`, `/staff/`).

# 3. **Fetch Linked Pages**:
#    - Collect all **outbound links** (pages the target page links to).
#    - Collect all **inbound links** (pages linking to the target page).

# ---

# ### **Step 2: Contextual Analysis of Linked Pages**
# 1. **Content Analysis**:
#    - Extract and analyze the content of linked pages for relevance to the six main categories or the "other" class.

# 2. **Categorization of Linked Pages**:
#    - Categorize each linked page using the same criteria applied to the target page (Step 1).
#    - Label linked pages as **main pages** or **resource pages**.

# ---

# ### **Step 3: Aggregated Feature Analysis**
# 1. **Contextual Consistency**:
#    - Check if the majority of linked pages belong to a specific category. For example:
#      - If most linked pages are categorized as faculty-related, the target page is likely faculty-related.

# 2. **Content-URL Alignment**:
#    - Combine content keywords and URL patterns to verify if the target page aligns with a specific category.

# 3. **Link Density & Type**:
#    - Assess whether the page has more inbound or outbound links to specific categories:
#      - A faculty page may have many inbound links from students and resource pages.
#      - A course page may have more outbound links to syllabus, assignments, or project pages.

# ---

# ### **Step 4: Decision Rule Application**
# 1. **Categorization Rules**:
#    - If the target page:
#      - **Predominantly links to faculty-related pages** or is linked to by them → **Classify as Faculty**.
#      - **Contains keywords like "course"** or links to syllabus/project resources → **Classify as Course**.
#      - **Features graduate student names/topics** and links to their resources → **Classify as Graduate Student**.
#      - **Contains department overview** or policy links → **Classify as Department**.
#      - **Relates to staff duties/contact information** → **Classify as Staff**.
#      - **Links to project descriptions** → **Classify as Project**.
#    - If it does not strongly align with any category and predominantly links to resource-type pages → **Classify as Other**.

# ---

# ### **Step 5: Validation**
# 1. **Cross-Check with Known Pages**:
#    - Validate the categorization by comparing the target page's profile with known categorized pages.

# 2. **Iterate if Necessary**:
#    - If ambiguity remains, refine content analysis or include additional context like link frequency or domain hierarchy.

# ---

# By systematically analyzing content, URL structure, and linked pages, this approach ensures accurate categorization based on both intrinsic and contextual features of the target webpage."""

    #refining_wisconsin=""

    refining_product="""Here's a step-by-step method to categorize the target product based on its name, description, and associated frequently purchased items:
    
    ---
    
    ### Step 1: **Analyze the Target Product**
    # 1. **Name Analysis**:
    #    - Extract key terms from the product name.
    #    - Look for direct category-related keywords.
    # 2. **Description Analysis**:
    #    - Identify functional details, intended use, or target audience.
    #    - Highlight adjectives or nouns that suggest a specific category (e.g., "organic," "children," "home use").
    #    
    # ---
    # 
    # ### Step 2: **Leverage Frequently Purchased Items**
    # - Identify shared features between the target product and the frequently purchased products (e.g., material, purpose, brand type).
    # - Cross-check which category best aligns with the characteristics of this group.
    # 
    ---
    
    ### Step 3: **Map to a Category**
    # 1. **Keyword Matching**:
    #    - Match keywords from the product name, description, and associated items to the defining features of each category.
    # 2. **Use Contextual Alignment**:
    #    - Consider the holistic context, e.g., if frequently purchased items and the product description suggest the product is for "office supplies," prioritize that category even if another seems partially relevant.
    # 
    # ---
    # 
    # ### Step 4: **Prioritize Category Specificity**
    # - If a product fits multiple categories:
    #    - Choose the most specific category.
    #    - If categories overlap heavily, default to the broader or parent category unless otherwise specified.
    # 
    ---
    
    ### Step 5: **Final Validation**
    # - Cross-check the selected category against edge cases or ambiguous categories.
    # - If uncertain, consider a second pass with a different perspective or criteria.
    # 
    # This structured approach ensures accuracy and consistency in categorizing products systematically."""
    refining_product="""To categorize a product systematically using its name, description, and frequently purchased-together items, follow these steps:

---

### **Step 1: Identify Primary Function and Context**
1. **Analyze the product name**: Look for keywords that describe the product's main function, purpose, or target audience.
   - Example: "Yoga Mat" suggests Sports & Outdoors.
2. **Review the product description**: Identify details about usage, materials, intended users, and associated environments.
   - Example: "Non-slip, durable, designed for exercise" confirms it aligns with Sports & Outdoors.

---

### **Step 2: Assess Frequently Purchased-Together Items**
1. **Examine associated products**:
   - Are they complementary or accessory items? Example: A yoga mat purchased with yoga blocks and resistance bands reinforces a Sports & Outdoors category.
   - Are they consumables, media, or equipment? This may narrow down categories (e.g., art supplies → Arts, Crafts & Sewing).

---

### **Step 3: Cross-Reference with Candidate Categories**
1. **Match keywords and usage to potential categories**:
   - If the product name includes "blender," cross-check with **Kitchen & Dining**, **Home & Kitchen**, or **Appliances**.
2. **Prioritize specific categories over general ones**:
   - **Home & Kitchen** is broad; if the product is explicitly for "Kitchen," prefer **Kitchen & Dining**.

---

### **Step 4: Eliminate Ambiguous or Incorrect Matches**
1. Exclude categories clearly unrelated to the product.
   - Example: A yoga mat wouldn’t belong in **Beauty** or **Electronics**.
2. If multiple categories apply, select the most specific:
   - Example: A blender for personal use aligns better with **Kitchen & Dining** than the broader **Home & Kitchen**.

---

### **Step 5: Finalize and Validate**
1. Confirm the selected category aligns with consumer expectations and typical product placements on Amazon.
2. Check if associated items reinforce the selected category.

---

### **Example**
#### Product: "Ceramic Coffee Mug"
1. **Analyze name and description**: "Ceramic" suggests material, "Coffee Mug" suggests usage in the kitchen or dining.
2. **Review frequently purchased items**: Matching items include coffee beans, coasters → supports Kitchen & Dining.
3. **Match with categories**: Specific match is **Kitchen & Dining**.
4. **Finalize**: Validate that the category fits expectations and listings for similar products.

---

Using this approach, you ensure systematic and precise categorization of Amazon products."""
    refining_product="""To categorize a target product from Amazon based on its name, description, and frequently purchased-together items, follow these steps:

### Step-by-Step Instructions:

1. **Analyze the Product Name:**
   - Identify keywords that hint at the product type, usage, or primary features.
   - For example, "Wireless Bluetooth Earbuds with Charging Case" suggests a category of "Electronics > Audio > Earbuds."

2. **Review the Product Description:**
   - Look for specific details like:
     - **Functionality**: What does the product do?
     - **Features**: Size, material, technology, or specific characteristics.
     - **Target Audience**: Adults, kids, professionals, or general use.
   - Highlight words that align with established categories (e.g., "smart home," "kitchen gadgets").

3. **Assess Frequently Purchased-Together Items:**
   - Examine the related products:
     - **Similar Products**: Suggests a specific subcategory (e.g., "other earbuds").
     - **Accessories**: Indicates the product's application (e.g., earbud cases suggest audio accessories).
     - **Complementary Products**: Contextualizes the category (e.g., chargers hint at electronics).

4. **Match with Amazon's Taxonomy:**
   - Use keywords from the name, description, and related products to match against Amazon’s known categories (e.g., "Home & Kitchen," "Electronics," "Beauty").
   - Cross-check the hierarchical structure to find the most specific fit (e.g., "Electronics > Wearable Technology > Smartwatches").

5. **Consider Usage Context:**
   - If the product serves multiple purposes, prioritize its primary function or the most common context indicated by the description and related items.

6. **Validate with Similar Products:**
   - Look at where similar products are categorized on Amazon for consistency.
   - Ensure your categorization aligns with user expectations and product discoverability.

### Example:
**Product Name:** "Stainless Steel Water Bottle with Straw Lid"
**Description:** "Eco-friendly, BPA-free, insulated bottle for hot and cold beverages, 32 oz."
**Frequently Purchased-Together Items:** Bottle brushes, silicone protective sleeves, replacement lids.

**Categorization Process:**
1. Keywords: "Water Bottle," "Insulated," "Eco-friendly."
2. Function: Beverage storage.
3. Accessories: Cleaning tools and lids imply drinkware.
4. Likely Category: "Home & Kitchen > Kitchen & Dining > Drinkware > Water Bottles."

By following these steps, you can systematically categorize any product using the provided information."""
    refining_product2="""To further revise the initial categorization of the target product and its frequently-purchased items, follow these steps:

### Step-by-Step Revision Instructions:

1. **Evaluate Category Specificity:**
   - Verify if the initial category is too broad or too narrow.
   - Check for subcategories that better fit the product's specific features or primary use.

2. **Analyze Alignment with Frequently-Purchased Items:**
   - Cross-check the categories of the frequently-purchased items.
   - Ensure the target product and its related items logically fit within a cohesive category or subcategories.

3. **Review Competitor Placement:**
   - Search for similar or competing products on Amazon.
   - Compare their categorization to identify trends or discrepancies.

4. **Examine Usage Context:**
   - Reassess the target product's primary use based on its description and frequently purchased items.
   - Adjust the category to reflect real-world application or user intent.

5. **Incorporate User Feedback and Search Patterns:**
   - If available, check customer reviews or Q&A sections for how users describe the product.
   - Use this language to validate or adjust the category for discoverability.

6. **Test Cross-Category Applicability:**
   - Determine if the product fits multiple categories (e.g., "Home & Kitchen" and "Sports & Outdoors").
   - Assign it to the most relevant primary category while considering a secondary categorization if applicable.

7. **Validate with Amazon's Guidelines:**
   - Review Amazon's categorization rules and guidelines for product listing.
   - Ensure compliance with their taxonomy and avoid misclassification.

8. **Check for Subcategory Conflicts:**
   - Verify that all frequently purchased items fit logically within the chosen subcategory structure.
   - Reassign if necessary to maintain consistency across the target product and its accessories.

### Example Revision:
**Initial Categorization:** "Home & Kitchen > Drinkware > Water Bottles."
**Frequently Purchased Items:** Brushes and protective sleeves categorized under "Kitchen Accessories."

**Revision Process:**
1. Check for more specific subcategories: Consider "Insulated Water Bottles."
2. Align related items: Match with "Accessories for Water Bottles."
3. Competitor check: Confirm similar products are under the same subcategory.
4. Final Categorization: "Sports & Outdoors > Hydration Gear > Water Bottles" (for outdoor use emphasis). 

This revision approach ensures the categorization is precise, user-relevant, and aligned with market trends."""
    refining_product21="""To further revise the initial categorization of the target product, use this systematic approach:

---

### Step-by-Step Instructions for Refinement:

1. **Review the Initial Categorization:**
   - Revisit the category chosen and the reasoning process.
   - Ensure the category aligns with the **product's primary purpose** and **user intent**.

2. **Reassess the Product Name and Description:**
   - Check if critical details in the name or description were overlooked or underweighted.
   - Look for niche-specific terms or attributes that suggest a more specific or alternate category.

3. **Reevaluate Frequently Purchased-Together Items:**
   - Identify patterns in the related items:
     - If items suggest broader or alternate usage, consider whether the product’s categorization should shift accordingly.
     - For example, complementary tools might indicate the product is part of a larger system or use case.

4. **Check for Category Conflicts:**
   - Analyze whether the product could fit into multiple categories:
     - Compare the current category against these alternatives.
     - Prioritize based on:
       - Primary function of the product.
       - Amazon’s category taxonomy and user search behavior.

5. **Refine Based on Usage Context:**
   - Use the combined information (name, description, related items) to pinpoint:
     - Core use cases (e.g., daily use, professional, niche hobby).
     - Specific consumer expectations (e.g., industrial vs. personal use).

6. **Consult Similar Products:**
   - Check the categorization of similar products or competitor listings.
   - If inconsistencies arise, refine the product's category to better align with common practices or expectations.

7. **Validate Against Amazon’s Taxonomy:**
   - Cross-check the refined category with Amazon’s official category hierarchy.
   - Ensure the revised categorization is specific yet inclusive enough for discoverability.

8. **Document Changes:**
   - Record the adjustments and reasons for the revised category.
   - Highlight how frequently purchased items, usage context, or overlooked details influenced the change.

---

### Example Application:

**Initial Categorization:** "Home & Kitchen > Kitchen & Dining > Drinkware > Water Bottles."

**Refinement Process:**
1. Product name suggests portability ("Stainless Steel Water Bottle").
2. Description emphasizes insulation, indicating use for both hot and cold beverages.
3. Related items (e.g., protective sleeves) highlight outdoor or active use.
4. Refine category to: **"Sports & Outdoors > Sports & Fitness > Hydration > Water Bottles"**, reflecting its utility for activities like hiking or gym use.

By iterating through these steps, the categorization becomes more accurate and aligned with customer intent and platform taxonomy."""

    refining_product2="""To refine the initial categorization of a target product after obtaining its initial category and reasoning, follow this approach:

### Step-by-Step Instructions:

1. **Verify Product Context:**
   - Revisit the product name and description to confirm the primary function and usage context.
   - Identify nuances such as multifunctionality, unique features, or niche applications that might suggest a narrower category.

2. **Reevaluate Frequently Purchased-Together Items:**
   - Cross-check related items for trends:
     - Do they emphasize a specific aspect of the product (e.g., accessories for portability, maintenance, or enhanced functionality)?
     - Are they primarily for complementary use or upgrades that hint at advanced applications?

3. **Analyze Competing Products:**
   - Research similar products on Amazon to:
     - Check if they are placed in the same category as your initial selection.
     - Identify patterns in subcategories or refinements (e.g., "Reusable Water Bottles" vs. "Sports Water Bottles").

4. **Incorporate Feedback from Amazon Taxonomy:**
   - Use Amazon's detailed category suggestions to match specific product attributes.
   - For example:
     - Look for subcategories focusing on specific features (e.g., insulated bottles under "Drinkware").
     - Check whether a niche category exists for a unique characteristic (e.g., "Eco-Friendly Drinkware").

5. **Adjust for Specificity or Breadth:**
   - If the product serves a specific purpose, refine to a more focused subcategory.
   - If it is multifunctional or has broad usage, consider categories that capture its versatility.

6. **Validate with Customer Intent:**
   - Consider where customers are most likely to search for or expect this product based on its purpose and features.
   - Align the category with customer behavior patterns (e.g., searching for a fitness water bottle under "Sports & Outdoors").

7. **Document the Refined Category:**
   - Clearly state the refined category and explain the reasoning for changes based on:
     - Unique product attributes.
     - Stronger alignment with related products.
     - Insights from Amazon taxonomy and customer expectations.

### Example:
**Initial Categorization:** "Home & Kitchen > Kitchen & Dining > Drinkware > Water Bottles."

**Refinement Process:**
1. The description emphasizes eco-friendliness and insulation.
2. Frequently purchased items (bottle brushes, protective sleeves) suggest portability and durability.
3. Competing products with similar features are listed under "Sports & Outdoors > Sports Water Bottles."
4. Adjust for specificity: "Sports & Outdoors > Sports Water Bottles > Insulated Water Bottles."

**Refined Category:** "Sports & Outdoors > Sports Water Bottles > Insulated Water Bottles" better reflects the product's features, related items, and customer intent. 

By systematically reevaluating each element, the refined category achieves greater precision and alignment with customer expectations."""
    refining_product2="""To refine the initial categorization of a target product from Amazon, follow these steps:

---

### **Step-by-Step Refinement Process:**

1. **Review the Initial Categorization:**
   - Consider the reasoning behind the initial category assignment.
   - Note any gaps or ambiguities in the reasoning that might need further clarification.

2. **Revisit the Product Name:**
   - Look for overlooked keywords that directly point to a more specific category or exclude certain categories.
   - Example: "GPS-enabled Smartwatch" might clarify a fit within "Electronics" over broader categories like "Sports & Outdoors."

3. **Reanalyze the Product Description:**
   - Search for any secondary functions or additional features.
   - Evaluate whether these suggest a category shift or refinement (e.g., a "fitness tracker" that includes sleep tracking might refine to "Electronics > Wearables").

4. **Cross-Check Frequently Purchased-Together Items:**
   - Identify patterns in related items:
     - Accessories for the same purpose suggest a match (e.g., cleaning supplies and lids for drinkware confirm "Home & Kitchen").
     - Disparate items may imply multiple potential use cases, warranting a closer look.
   - Validate consistency with the chosen category (e.g., tools purchased with electronics might clarify "Tools & Home Improvement").

5. **Match Against the Candidate Categories:**
   - Compare the initial choice to the candidate list:
     - Exclude unrelated categories using process of elimination.
     - Narrow down the most fitting subcategories (e.g., if the initial choice is "Home & Kitchen," refine to "Kitchen & Dining").

6. **Refine Based on Specificity:**
   - Prioritize the most specific applicable category available in the list.
   - Use a hierarchy to ensure the product is not over-generalized (e.g., "Home & Kitchen > Kitchen & Dining > Drinkware").

7. **Double-Check Complementary Categories:**
   - If the product overlaps with multiple categories, choose the one that best aligns with its **primary use case**.
   - Example: A "stainless steel water bottle" may fit both "Sports & Outdoors" and "Home & Kitchen," but its description and related items might confirm "Home & Kitchen > Kitchen & Dining > Drinkware."

8. **Validate Against Customer Expectations:**
   - Ensure the category aligns with where customers are likely to look for the product.
   - Refer to similar products for precedent (e.g., browse Amazon for comparable items).

---

### Example:

**Initial Categorization:** "Home & Kitchen"  
**Reasoning:** Based on product description and related items, it seems to be used primarily in the home.  

**Refinement Process:**
1. Name and description mention "insulated bottle" for beverages.
2. Frequently purchased-together items include lids and cleaning tools for drinkware.
3. Candidate Categories:
   - "Home & Kitchen" → Refined to "Kitchen & Dining > Drinkware > Water Bottles."
   - Exclude "Sports & Outdoors" (no sports-specific features emphasized).

**Refined Category:** "Home & Kitchen > Kitchen & Dining > Drinkware > Water Bottles."

By iterating through this process, you can systematically refine the categorization."""
    refining_product2="""To categorize a target product given its name, description, initial categorization, reasons, and frequently purchased-together items, follow these systematic steps:

---

### Step-by-Step Instructions:

#### **1. Understand the Target Product:**
   - **Analyze Name:** Identify keywords indicating product type, usage, or primary features.
   - **Examine Description:** Look for specific functions, features, target audience, and context.
   - **Review Initial Categorization:** Note the suggested category and reasons to understand the rationale.
   - **Highlight Gaps:** Identify missing or unclear information that could refine categorization.

#### **2. Analyze Frequently Purchased-Together Items:**
   - **Inspect Names:** Extract relevant keywords to identify similarities or complementary purposes.
   - **Review Descriptions:** Determine the function or category of each item to establish contextual relationships.
   - **Cross-check Initial Categorization:** Ensure related items' categorization aligns logically with the target product.

#### **3. Determine Primary Function and Context:**
   - Combine insights from the target product's details and related items to deduce the main use case.
   - Consider complementary items as an indicator of the broader context (e.g., an accessory for a specific type of product).

#### **4. Validate Against Amazon's Taxonomy:**
   - Cross-reference keywords and functions with Amazon's hierarchical categories.
   - Ensure consistency by checking where similar products and frequently purchased-together items are categorized.

#### **5. Prioritize Precision:**
   - Select the **most specific category** that describes the product and its primary purpose.
   - Ensure that the categorization aligns with the product's description, usage context, and related items.

#### **6. Refine Based on Insights:**
   - Reassess the initial categorization and reasons in light of the gathered evidence.
   - Adjust categorization if the frequently purchased-together items suggest a different or more precise category.

---

### Example:

**Target Product:**
- **Name:** "Ergonomic Office Chair with Lumbar Support"
- **Description:** "Adjustable height, mesh backrest, rolling casters, ideal for home or office use."
- **Initial Categorization:** "Furniture > Office Furniture > Chairs"
- **Reasons:** "Designed for ergonomic support, suitable for office settings."

**Frequently Purchased-Together Items:**
1. **Name:** "Memory Foam Seat Cushion"
   - **Description:** "Enhances comfort for prolonged sitting."
   - **Initial Categorization:** "Home & Kitchen > Bedding > Cushions."
2. **Name:** "Adjustable Footrest for Office Chairs"
   - **Description:** "Improves leg posture while seated."
   - **Initial Categorization:** "Furniture > Office Furniture > Accessories."

**Categorization Process:**
1. **Target Product Analysis:**
   - Keywords: "Office Chair," "Ergonomic," "Lumbar Support."
   - Function: Office seating.
   - Initial categorization is logical but could benefit from precision if context indicates a subcategory.

2. **Related Items Context:**
   - Complementary: Items enhance office seating comfort and ergonomics.
   - Aligns with office-focused furniture.

3. **Refinement:**
   - Cross-check with Amazon taxonomy for a more precise subcategory: “Furniture > Office Furniture > Ergonomic Chairs.”

4. **Final Categorization:**
   - Adjust to reflect refined insights if needed, ensuring consistency across related products.

By following these steps, categorization becomes systematic and contextually accurate."""
    refining_product22="""Here's a **systematic step-by-step instruction** to categorize the target product effectively, given the specified inputs:

---

### Step 1: **Understand the Target Product**
1. **Analyze the Name and Description:**
   - Identify primary keywords that describe the product's core functionality, features, and target audience.
   - Note any specific phrases indicating the product's purpose or usage context.

2. **Evaluate the Initial Categorization:**
   - Consider the rationale behind the initial categorization.
   - Assess whether it aligns with the product's primary function and features.

---

### Step 2: **Analyze Frequently Purchased-Together Items**
1. **Extract Key Information:**
   - Review the names and descriptions of frequently purchased-together items.
   - Highlight features or uses that are complementary or directly related to the target product.

2. **Compare Initial Categorization of Related Items:**
   - Check if their initial categorization supports or conflicts with the target product’s categorization.
   - Identify patterns in their placement within the candidate categories.

3. **Determine Context and Application:**
   - If most related items fall within a specific category (e.g., Kitchen & Dining, Sports & Outdoors), this indicates the likely ecosystem for the target product.

---

### Step 3: **Reassess Initial Categorization**
1. **Check Against Candidate Categories:**
   - Match the product and its context (including related items) against the candidate category list.
   - Focus on categories reflecting primary use, consumer intent, and Amazon taxonomy norms.

2. **Eliminate Overlapping or Incorrect Categories:**
   - Disregard categories that do not align with the target product's specific function or context.
   - For example, if the product is a kitchen gadget, exclude categories like “Electronics” or “Sports & Outdoors” unless clearly relevant.

---

### Step 4: **Finalize Categorization**
1. **Prioritize the Primary Use Case:**
   - Place the product in the category that best reflects its primary purpose.
   - Use the related items' categorization to validate or refine the decision.

2. **Subcategorization Consideration:**
   - If the product could fit into a broader category (e.g., Home & Kitchen), consider whether a more specific subcategory (e.g., Kitchen & Dining) provides better precision.

3. **Cross-Check with Amazon Taxonomy:**
   - Ensure consistency with similar products and their placement on Amazon.

---

### Example Walkthrough:

#### **Inputs:**
- **Target Product Name:** "Electric Coffee Grinder"
- **Description:** "Compact, stainless steel grinder for coffee beans, spices, and herbs."
- **Initial Categorization:** "Kitchen & Dining"
- **Reason:** Used primarily for food preparation.
- **Frequently Purchased-Together Items:** 
  - **Item 1:** "Coffee Beans" (Category: Grocery & Gourmet Food)
  - **Item 2:** "Coffee Grinder Brush" (Category: Kitchen & Dining)
  - **Item 3:** "Reusable Coffee Filters" (Category: Home & Kitchen)

#### **Steps Applied:**
1. **Analyze Name/Description:**
   - Keywords: "Coffee Grinder," "Food Preparation."
2. **Evaluate Related Items:**
   - Related to coffee preparation (Coffee Beans, Filters).
   - Frequent items categorized in “Kitchen & Dining” and “Home & Kitchen.”
3. **Reassess Categorization:**
   - Initial category aligns; no strong reason to move to broader or unrelated categories.
4. **Finalize:**
   - Category: "Kitchen & Dining."

--- 

### Outcome:
By systematically analyzing the target product and related items, you ensure its categorization aligns with both functionality and consumer intent."""
    refining_product3="""To categorize a target Amazon product effectively using its name, description, initial categorization and reasons, along with its frequently purchased-together items (each with their own name, description, initial categorization, and reasons), follow these systematic steps:

### Step-by-Step Instructions:

1. **Gather All Relevant Information:**
   - **Target Product:**
     - **Name**
     - **Description**
     - **Initial Categorization**
     - **Reasons for Initial Categorization**
   - **Frequently Purchased-Together Items:**
     - For each item, collect:
       - **Name**
       - **Description**
       - **Initial Categorization**
       - **Reasons for Initial Categorization**

2. **Analyze the Target Product:**
   - **Identify Key Attributes:**
     - Extract primary keywords related to the product’s function, features, and intended use from the name and description.
   - **Understand Initial Categorization:**
     - Review the reasons provided to comprehend the rationale behind the current category placement.

3. **Examine Frequently Purchased-Together Items:**
   - **Identify Common Themes:**
     - Look for overlapping categories, complementary functions, or shared features among the related items.
   - **Assess Relevance:**
     - Determine how each related item's categorization supports or suggests a different categorization for the target product.

4. **Map to Amazon’s Taxonomy:**
   - **Navigate Amazon’s Category Structure:**
     - Use the identified keywords and themes to explore Amazon’s official categories and subcategories.
   - **Find the Most Specific Fit:**
     - Aim for the most precise category that accurately reflects the product’s primary purpose and attributes.

5. **Integrate Insights from Related Items:**
   - **Align with Complementary Products:**
     - Ensure that the target product’s category logically fits with the categories of its frequently purchased-together items.
   - **Adjust if Necessary:**
     - If related items consistently fall under a specific subcategory, consider aligning the target product accordingly for better coherence.

6. **Evaluate and Refine the Initial Categorization:**
   - **Compare Categories:**
     - Contrast the initial categorization with the insights gained from analyzing related items and Amazon’s taxonomy.
   - **Make Adjustments:**
     - Modify the category if the initial one is too broad, too narrow, or misaligned based on the comprehensive analysis.

7. **Validate with Similar Products:**
   - **Research Comparable Listings:**
     - Look at how similar products are categorized on Amazon to ensure consistency and discoverability.
   - **Ensure User Alignment:**
     - Confirm that the categorization matches user expectations and common search behaviors.

8. **Finalize the Categorization:**
   - **Assign the Category:**
     - Place the target product into the most appropriate and specific category identified.
   - **Document the Rationale:**
     - Note the reasons for the chosen category to maintain clarity and for future reference.

### Example Workflow:

**Target Product:**
- **Name:** "Stainless Steel Insulated Travel Mug"
- **Description:** "Keeps beverages hot for 12 hours and cold for 24 hours, leak-proof lid, BPA-free."
- **Initial Categorization:** "Home & Kitchen > Drinkware > Mugs"
- **Reasons:** Durable material, suitable for travel and daily use.

**Frequently Purchased-Together Items:**
1. **Name:** "Travel Mug Cleaning Brush"
   - **Description:** "Flexible brush for cleaning narrow bottle openings."
   - **Initial Categorization:** "Home & Kitchen > Cleaning Tools"
   - **Reasons:** Essential for maintaining the travel mug.

2. **Name:** "Silicone Travel Lid Replacement"
   - **Description:** "Spill-resistant silicone lid compatible with various travel mugs."
   - **Initial Categorization:** "Home & Kitchen > Drinkware Accessories"
   - **Reasons:** Replacement part for the travel mug.

**Categorization Process:**
1. **Identify Key Attributes:** Insulated, travel-friendly, leak-proof, BPA-free.
2. **Analyze Related Items:** Cleaning tools and accessories related to drinkware.
3. **Map to Amazon’s Taxonomy:** "Home & Kitchen > Drinkware > Travel Mugs & Tumblers."
4. **Align with Related Items:** Ensure accessories fall under the same or compatible subcategories.
5. **Refine Initial Categorization:** Move from "Mugs" to the more specific "Travel Mugs & Tumblers" for precision.
6. **Validate:** Similar products are listed under "Travel Mugs & Tumblers."
7. **Finalize:** Categorize the product as "Home & Kitchen > Drinkware > Travel Mugs & Tumblers."

By following these steps, you ensure that the product is categorized accurately, enhancing its visibility and alignment with customer expectations."""
    refining_product31="""To categorize a target Amazon product effectively using its name, description, initial categorization and reasons, along with its frequently purchased-together items (each with their own name, description, initial categorization, and reasons), follow these systematic steps:

### Step-by-Step Instructions:

1. **Gather All Relevant Information:**
   - **Target Product:**
     - **Name**
     - **Description**
     - **Initial Categorization**
     - **Reasons for Initial Categorization**
   - **Frequently Purchased-Together Items:**
     - For each item, collect:
       - **Name**
       - **Description**
       - **Initial Categorization**
       - **Reasons for Initial Categorization**

2. **Analyze the Target Product:**
   - **Identify Key Attributes:**
     - Extract primary keywords related to the product’s function, features, and intended use from the name and description.
   - **Understand Initial Categorization:**
     - Review the reasons provided to comprehend the rationale behind the current category placement.

3. **Examine Frequently Purchased-Together Items:**
   - **Identify Common Themes:**
     - Look for overlapping categories, complementary functions, or shared features among the related items.
   - **Assess Relevance:**
     - Determine how each related item's categorization supports or suggests a different categorization for the target product.

4. **Map to Amazon’s Taxonomy:**
   - **Navigate Amazon’s Category Structure:**
     - Use the identified keywords and themes to explore Amazon’s official categories and subcategories.
   - **Find the Most Specific Fit:**
     - Aim for the most precise category that accurately reflects the product’s primary purpose and attributes.

5. **Integrate Insights from Related Items:**
   - **Align with Complementary Products:**
     - Ensure that the target product’s category logically fits with the categories of its frequently purchased-together items.
   - **Adjust if Necessary:**
     - If related items consistently fall under a specific subcategory, consider aligning the target product accordingly for better coherence.

6. **Evaluate and Refine the Initial Categorization:**
   - **Compare Categories:**
     - Contrast the initial categorization with the insights gained from analyzing related items and Amazon’s taxonomy.
   - **Make Adjustments:**
     - Modify the category if the initial one is too broad, too narrow, or misaligned based on the comprehensive analysis.

7. **Validate with Similar Products:**
   - **Research Comparable Listings:**
     - Look at how similar products are categorized on Amazon to ensure consistency and discoverability.
   - **Ensure User Alignment:**
     - Confirm that the categorization matches user expectations and common search behaviors.

8. **Finalize the Categorization:**
   - **Assign the Category:**
     - Place the target product into the most appropriate and specific category identified.
   - **Document the Rationale:**
     - Note the reasons for the chosen category to maintain clarity and for future reference.

By following these steps, you ensure that the product is categorized accurately, enhancing its visibility and alignment with customer expectations."""
    refining_product32="""To categorize a target Amazon product effectively using its name, description, initial categorization and reasons, along with its frequently purchased-together items (each with their own name, description, initial categorization, and reasons), follow these systematic steps:

    ### Example Workflow:

**Target Product:**
- **Name:** "Stainless Steel Insulated Travel Mug"
- **Description:** "Keeps beverages hot for 12 hours and cold for 24 hours, leak-proof lid, BPA-free."
- **Initial Categorization:** "Home & Kitchen > Drinkware > Mugs"
- **Reasons:** Durable material, suitable for travel and daily use.

**Frequently Purchased-Together Items:**
1. **Name:** "Travel Mug Cleaning Brush"
   - **Description:** "Flexible brush for cleaning narrow bottle openings."
   - **Initial Categorization:** "Home & Kitchen > Cleaning Tools"
   - **Reasons:** Essential for maintaining the travel mug.

2. **Name:** "Silicone Travel Lid Replacement"
   - **Description:** "Spill-resistant silicone lid compatible with various travel mugs."
   - **Initial Categorization:** "Home & Kitchen > Drinkware Accessories"
   - **Reasons:** Replacement part for the travel mug.

**Categorization Process:**
1. **Identify Key Attributes:** Insulated, travel-friendly, leak-proof, BPA-free.
2. **Analyze Related Items:** Cleaning tools and accessories related to drinkware.
3. **Map to Amazon’s Taxonomy:** "Home & Kitchen > Drinkware > Travel Mugs & Tumblers."
4. **Align with Related Items:** Ensure accessories fall under the same or compatible subcategories.
5. **Refine Initial Categorization:** Move from "Mugs" to the more specific "Travel Mugs & Tumblers" for precision.
6. **Validate:** Similar products are listed under "Travel Mugs & Tumblers."
7. **Finalize:** Categorize the product as "Home & Kitchen > Drinkware > Travel Mugs & Tumblers."

By following these steps, you ensure that the product is categorized accurately, enhancing its visibility and alignment with customer expectations.""" 
    refining_product4="""### Step-by-Step Instructions to Refine the Initial Categorization:

1. **Review the Initial Categorization and Reasoning:**
   - Revisit the initial category assignment.
   - Evaluate the reasoning provided—was it based on strong keywords, functionality, or related products?
   - Identify any gaps or assumptions that need clarification.

2. **Reassess the Product Name:**
   - Reanalyze for specific terms that may have been overlooked or misinterpreted in the initial categorization.
   - Look for new insights or alternate meanings (e.g., "portable" could imply travel-related categories).

3. **Examine the Product Description in Detail:**
   - Search for secondary features, usage scenarios, or implied contexts that were not prioritized initially.
   - Determine whether the product has overlapping use cases (e.g., "kitchen" vs. "outdoor").

4. **Evaluate Frequently Purchased-Together Items:**
   - Categorize the related items into:
     - **Similar Products**: Reinforces the initial categorization if consistent.
     - **Accessories**: Suggests refinements based on primary use or subcategory (e.g., cases for electronic devices).
     - **Complementary Items**: Indicates specific applications or user contexts (e.g., camping gear paired with water bottles suggests outdoor use).

5. **Cross-Check Against Amazon Taxonomy:**
   - Revisit Amazon’s category hierarchy to validate if a more specific or accurate subcategory exists.
   - Look for keywords or contexts implied by related items (e.g., "travel" vs. "home").

6. **Prioritize Based on Product Use Context:**
   - Determine the primary function or setting where the product is most commonly used.
   - Adjust the category to reflect the product's core purpose or dominant use case.

7. **Benchmark Against Similar Products:**
   - Search for comparable products on Amazon and note their categorization.
   - If the majority are in a different category than initially chosen, refine accordingly.

8. **Validate the Final Categorization:**
   - Confirm that the final category aligns with:
     - Product discoverability.
     - User expectations based on the product's features and related items.

### Example:

**Initial Categorization:** "Home & Kitchen > Kitchen & Dining > Drinkware > Water Bottles"  
**Reasons:** Keywords "water bottle" and "drinkware."  
**Frequently Purchased-Together Items:** Insulated lunch bags, carabiner clips, sports backpacks.  

**Refinement Process:**
1. Name and Description emphasize portability and insulation for outdoor use.
2. Related items suggest outdoor or sports context.
3. Refined Category: "Sports & Outdoors > Outdoor Recreation > Hydration > Water Bottles."

By following this systematic approach, you ensure a refined, accurate, and context-sensitive categorization."""
    refining_product5="""### Step-by-Step Instructions to Revise and Improve Product Categorization:

1. **Understand the Initial Categorization:**
   - **Review the Name and Description:** Extract key details about the product's type, purpose, features, and target audience.
   - **Analyze the Initial Categorization:** Note the chosen category and the subordinate’s reasoning. Identify keywords they relied on.

2. **Evaluate Frequently Purchased-Together Items:**
   - Assess how these items relate to the target product:
     - **Accessories:** Confirm compatibility (e.g., protective cases for phones).
     - **Complementary Products:** Verify usage context (e.g., cleaning brushes for water bottles suggest drinkware).
   - Ensure the relationship supports the product's primary purpose and not secondary uses.

3. **Check for Logical and Factual Errors:**
   - Compare the initial category against the product's:
     - **Primary Functionality:** Ensure the category reflects the main use (e.g., a water bottle categorized under "Outdoor Gear" instead of "Drinkware" might indicate confusion).
     - **Target Audience:** Verify the match between the intended audience and the category.
   - Look for **deviations in definition:** Cross-check the category’s description on Amazon or similar platforms to ensure alignment.

4. **Cross-Reference with Similar Products:**
   - Search for products with similar names and descriptions on Amazon.
   - Identify their categorization and ensure consistency, adjusting for misaligned examples.

5. **Reassess Hierarchical Fit:**
   - Confirm the product fits within the parent and subcategories:
     - **Example:** "Home & Kitchen > Kitchen & Dining > Drinkware > Water Bottles" is more appropriate than a broad category like "Sports & Outdoors" unless specifically marketed for athletic use.

6. **Consider Customer Intent and Search Behavior:**
   - Think about where customers are most likely to look for the product.
   - Prioritize the category that maximizes discoverability and relevance.

7. **Make Adjustments and Provide Justification:**
   - Suggest a corrected category if needed.
   - Provide a concise explanation, citing evidence from the product’s name, description, and related items.

### Example of Revised Categorization:
**Product:** "Stainless Steel Water Bottle with Straw Lid"
**Initial Categorization by Subordinate:** "Sports & Outdoors > Outdoor Recreation > Hydration."
**Reason Provided:** "Frequently used for outdoor activities."

**Revised Categorization:**
- Correct Category: "Home & Kitchen > Kitchen & Dining > Drinkware > Water Bottles."
- Justification:
  1. The product description emphasizes versatility (hot and cold beverages), aligning with general drinkware.
  2. Frequently purchased-together items (bottle brushes, replacement lids) are commonly found with drinkware, not exclusively outdoor gear.
  3. Cross-checking similar products on Amazon confirms most stainless steel water bottles are categorized under drinkware.

This systematic approach ensures logical, factual, and customer-focused categorization."""
    refining_product5="""Your task is to select the single most appropriate category for a target Amazon product from the provided candidate list. Follow this structured process:

---

## Step 1: Identify the Product's Primary Function

- Extract the **core purpose** of the product from its name and description (e.g., is it something you wear, eat, read, install, play with, use in a kitchen, etc.).
- Determine the **primary consumer context**: home use, professional/industrial use, personal care, entertainment, outdoors, automotive, etc.
- Do NOT be misled by secondary features (e.g., a "smart" scale is still a "Health & Personal Care" item, not "Electronics").

---

## Step 2: Analyze the Co-Purchase Network

Frequently purchased-together items reveal the product's **ecosystem** — the real-world context in which it is used. Apply the following logic:

- If co-purchased items **share a consistent category** (e.g., all are sports equipment), this strongly confirms the target product belongs to the same category.
- If co-purchased items are **accessories or consumables** for the target product (e.g., replacement filters, cleaning brushes, chargers), identify what main product they serve — that points to the correct category.
- If co-purchased items span **multiple categories**, weight those that are most functionally related to the target product's primary use.
- A product consistently co-purchased with items from category X almost certainly belongs to category X or a closely adjacent one.

---

## Step 3: Resolve Confusable Category Pairs

Many categories in the list are easily confused. Apply these disambiguation rules:

**Electronics cluster:**
- `Electronics`: General consumer electronics (TVs, speakers, projectors, general gadgets).
- `All Electronics`: Broad electronics catch-all; prefer a more specific electronics category when possible.
- `Computers`: Laptops, desktops, monitors, RAM, SSDs, keyboards, mice — computing hardware and peripherals.
- `Cell Phones & Accessories`: Smartphones, phone cases, screen protectors, chargers specifically for mobile phones.
- `Camera & Photo`: Cameras, lenses, tripods, memory cards for photography/video.
- `Car Electronics`: GPS devices, dash cams, car audio, in-vehicle electronic accessories.
- `MP3 Players & Accessories`: Dedicated portable music players and their accessories.
- `GPS & Navigation`: Standalone GPS devices and navigation accessories (non-automotive context).

**Home & Living cluster:**
- `Home & Kitchen`: Broad category for home goods, cleaning, bedding, décor.
- `Kitchen & Dining`: More specific — cookware, bakeware, tableware, small kitchen appliances, utensils.
- `Appliances`: Large home appliances (washing machines, refrigerators, dishwashers, ovens).
- `Tools & Home Improvement`: Power tools, hand tools, hardware, plumbing, electrical, renovation supplies.
- `Home Improvement`: Specifically for renovation/repair materials and fixtures (overlaps with Tools & Home Improvement; prefer `Tools & Home Improvement` for tools).
- `Furniture & Decor`: Furniture items (chairs, tables, shelves) and decorative home items.
- `Patio, Lawn & Garden`: Outdoor home items — garden tools, lawn mowers, patio furniture, planters.

**Personal care & fashion cluster:**
- `Health & Personal Care`: Medical devices, vitamins, supplements, personal hygiene products, first-aid items.
- `Beauty`: Cosmetics, makeup, skincare, haircare products marketed as mainstream beauty.
- `All Beauty`: Broad beauty catch-all; prefer `Beauty` or `Luxury Beauty` when applicable.
- `Luxury Beauty`: Premium/high-end beauty brands.
- `Clothing, Shoes & Jewelry`: Apparel, footwear, watches, handbags, jewelry.
- `Amazon Fashion`: Amazon's own fashion line; prefer `Clothing, Shoes & Jewelry` for general fashion items.
- `Baby Products` / `Baby`: Items specifically designed for infants and toddlers (diapers, baby food, infant toys).

**Entertainment & media cluster:**
- `Books`: Physical printed books.
- `Kindle Store` / `Buy a Kindle`: Kindle devices and digital book purchases — not physical books.
- `Digital Music`: Downloadable/streaming music products.
- `CDs & Vinyl`: Physical music media.
- `Movies & TV`: DVDs, Blu-rays, streaming content.
- `Video Games`: Game software, gaming consoles, controllers.
- `Software`: PC/Mac software applications.
- `Magazine Subscriptions`: Periodical subscriptions.

**Other distinct categories:**
- `Sports & Outdoors`: Fitness equipment, outdoor recreation gear, camping supplies, athletic clothing/shoes.
- `Toys & Games`: Children's toys, board games, puzzles (non-video games).
- `Automotive`: Car parts, car care, maintenance tools for vehicles (non-electronic).
- `Pet Supplies`: Food, accessories, health products for pets.
- `Grocery & Gourmet Food`: Edible products, beverages, cooking ingredients.
- `Arts, Crafts & Sewing`: Art supplies, craft materials, sewing notions, fabric.
- `Office Products`: Office furniture, business supplies, paper products, printers.
- `Industrial & Scientific`: Lab equipment, safety gear, industrial tools, scientific instruments.
- `Musical Instruments`: Instruments, amplifiers, music accessories.
- `Collectibles & Fine Art`: Rare or collectable items, artwork, memorabilia.

**Special/noise labels — avoid unless overwhelming evidence:**
- `label 25`, `#508510`, `Purchase Circles`, `Gift Cards`: These are data artifacts or highly niche categories. Only select them if all other categories clearly do not fit.

---

## Step 4: Make a Decision

1. Eliminate categories that clearly do not match the product's primary function or consumer context.
2. From the remaining candidates, select the **most specific** category that still captures the product's primary purpose.
3. Use the co-purchase network as a **tie-breaker**: if two categories are equally plausible, prefer the one that aligns with the majority of co-purchased items.
4. Your final answer must be **exactly one category name** from the provided list — do not invent subcategories or hierarchical paths.

---

## Step 5: Final Self-Check

Before finalising, verify:
- Does the chosen category reflect the product's **primary** use, not a secondary or edge-case use?
- Is the chosen category consistent with the **co-purchase ecosystem**?
- Have you avoided confusable neighbouring categories by applying the disambiguation rules above?
- Is the chosen category name **exactly as it appears** in the candidate list?

If any check fails, revise your answer accordingly."""
    refining_product5="""**System Prompt:**
As an expert product categorizer, your task is to select the single most appropriate Amazon product category for a target product. This dataset is an **interconnected co-purchase graph**. You must balance the product's fundamental physical/functional attributes with the topological context provided by "Frequently purchased-together items".

Please strictly follow this **Waterfall Decision Process**:

---

## Step 1: Core Function & Noise Filtration
- **Identify Core Purpose:** Determine what the item actually *does*. Do NOT be misled by secondary "smart" features.
- **Avoid Data Artifacts (CRITICAL):** Absolutely DO NOT select `label 25`, `#508510`, `Purchase Circles`, or `Gift Cards`. If the ground truth happens to be one of these, it is a dataset error. You must still predict the true logical category.
- **Missing Description:** If the product description is missing or vague, rely 100% on the co-purchased items to determine the category.

---

## Step 2: The Hub-Node Immunity (Absolute Priority 1)
- **Rule:** Universal electronic consumables (e.g., SD cards, AA/AAA batteries, generic USB drives) are IMMUNE to the graph ecosystem. An SD card is ALWAYS `Electronics` or `Computers`.

---

## Step 3: Hard-Boundary Vetoes & Specialized Domains (Absolute Priority 2)
Amazon taxonomy overrides graph topology based on physical form. Apply these vetoes IMMEDIATELY:
- **The OPE Veto (Outdoor Power Equipment):** Chainsaws, lawnmowers, and their specific accessories/protective apparel (e.g., Husqvarna chaps/helmets) MUST go to `Patio, Lawn & Garden` (NOT Tools or Clothing).
- **The Installation & Fixture Veto:** Hardwired items (wall plates, built-in fixtures, permanent lighting) MUST go to `Tools & Home Improvement`. *(Exception: TV wall mounts/brackets go to `Electronics` per the Host-Device rule).*
- **The Painting & Color Veto:** Pantone color guides and paint swatches are painting tools and MUST go to `Tools & Home Improvement`.
- **The Appliance Veto:** Humidifiers, air purifiers, and heaters (even if marketed specifically for pets or terrariums) MUST go to `Home & Kitchen` (or Appliances).
- **The Apparel & Luggage Veto:** Everyday wearables AND everyday luggage (backpacks, suitcases) MUST be `Clothing, Shoes & Jewelry`. *(Exceptions: Chainsaw chaps -> Patio; Party masks -> Toys).*
- **The Toys, Party & Kids' Crafts Veto:** Action figures, Anime PVC statues, party supplies, festive decorations, AND Kids' Arts & Crafts (Crayola, coloring books) MUST go to `Toys & Games`. (Leave `Collectibles & Fine Art` ONLY for rare coins and signed memorabilia).
- **The Educational Veto:** Classroom decorations and teacher supplies MUST go to `Office Products`.
- **The Furniture & Wall Art Veto:** Posters, prints, wall art, or large storage furniture MUST default to `Home & Kitchen`. Ignore small co-purchased pens/tools.
- **The Pro-Audio Veto:** Professional studio/stage audio gear (standalone XLR microphones, mixers) MUST go to `Musical Instruments`. *(Exception: Audio adapters, DI boxes, and generic A/V cables stay in `Electronics`).*
- **The Optics & Photography Veto:** Telescopes and binoculars belong to `Camera & Photo`. However, Studio Lighting Kits, Strobes, and Softboxes map to `Electronics`.
- **The Physical Media Veto:** Physical books -> `Books`. Music albums/bands/tracks -> `CDs & Vinyl`. Workout DVDs/Films -> `Movies & TV`. Digital books -> `Kindle Store`.
- **The Edible & Supplement Veto:** Whole foods or natural snacks MUST be `Grocery & Gourmet Food`. *(Exception: Superfoods like cacao nibs co-purchased with vitamins MUST go to `Health & Personal Care`).*

---

## Step 4: The Ultimate Subordination Rule & Graph Inference (The Default Engine)
In Amazon's taxonomy, the **Target of Service (Host)** ALWAYS overrides the physical environment or form:
- **If the Host is a Device (Phones/TVs/Cameras):** The accessory inherits the device's category. (e.g., Cell phone car mount -> `Cell Phones & Accessories`; TV Wall Mount -> `Electronics`).
- **If the Host is a Vehicle (Cars/Motorcycles):** The tool, part, or fluid inherits the vehicle's category. (e.g., Car wax -> `Automotive`).
- **Absolute Graph Submission for Generic Materials:** Generic raw materials (e.g., elastic cord) are 100% dictated by their graph neighbors.
- **Domain over Form:** Generic functional items inherit graph neighbors. Tape and adhesives (even "craft tape") usually default to `Tools & Home Improvement`.
- **Ignore Cart-Fillers:** Disregard obvious random co-purchases like snacks/cookies.

---

## Step 5: Resolve Amazon-Specific Traps & Decoys
1. **Household Supplies:** Laundry detergents, dish soaps, AND disposable tableware (plastic cups, paper plates, toilet paper) MUST go to `Health & Personal Care` (NOT Home & Kitchen).
2. **Skin & Lip Care:** Sunscreens and facial cleansers go to `Beauty`. However, medicated Lip Balms/Protectants go to `Health & Personal Care`.
3. **Specialized Outdoor First-Aid:** General first-aid is Health, but Backpacking/Camping-specific survival kits go to `Sports & Outdoors`.
4. **Casino & Game Room:** Casino equipment and card shufflers belong to `Sports & Outdoors`.
5. **Industrial & 3D Printing:** 3D printer filaments and heavy commercial supplies go to `Industrial & Scientific`.
6. **Cable Management & PC Components:** Cord protectors, wire raceways, internal Hard Drives (HDD/SSD), laptop screens, and bare internal components map to `Electronics` (NOT Computers).
7. **Coolants vs. First Aid:** Generic ice packs for food belong to `Home & Kitchen`.
8. **Bath & Body:** Artisanal soaps, body washes, and bath sponges -> `Beauty`.
9. **Decoy Label Avoidance (CRITICAL):** - ALWAYS prefer `Home & Kitchen` for cookware/bakeware. **NEVER use `Kitchen & Dining` or `Furniture & Decor`.**
   - ALWAYS prefer `Office Products`. **NEVER use `Office & School Supplies`.**
   - ALWAYS prefer `Baby Products`. **NEVER use `Baby`.**

---

## Step 6: Master Sanity Checklist (MENTAL VERIFICATION)
Before outputting, mentally verify against these structured trap-checks:

**[ ] A. Lifestyle & Fashion Traps:**
- Backpack/Luggage? -> FORCE `Clothing, Shoes & Jewelry`.
- Kids' Craft/Crayola/Party supply? -> FORCE `Toys & Games`.
- Casino/Poker equipment? -> FORCE `Sports & Outdoors`.

**[ ] B. Home & Hardware Traps:**
- Cookware/Bakeware/Ice Packs? -> FORCE `Home & Kitchen` (Avoid Kitchen & Dining).
- Hardwired fixture or Pantone guide? -> FORCE `Tools & Home Improvement`.
- Humidifier/Heater? -> FORCE `Home & Kitchen`.

**[ ] C. Tech & Media Traps:**
- TV Mount / Studio Light Kit / Audio Cable / HDD? -> FORCE `Electronics`.
- Pro XLR Mic / Audio Mixer? -> FORCE `Musical Instruments`.
- Music Album/Book/DVD? -> Match specific Physical Media category.

**[ ] D. Health & Outdoors Traps:**
- Plastic Cups / Toilet Paper / Detergent? -> FORCE `Health & Personal Care`.
- Lip Balm (medicated) / Superfoods+Vitamins? -> FORCE `Health & Personal Care`.
- Chainsaw gear / Outdoor Power Equipment? -> FORCE `Patio, Lawn & Garden`.

**[ ] E. Graph & Decoy Traps:**
- Generic material (e.g., elastic cord)? -> MUST follow graph neighbors.
- Did I pick a Decoy (`Office & School Supplies`, `Kitchen & Dining`, `#508510`)? -> CHANGE IT.
"""
    refining_product5="""**System Prompt:**
As an expert product categorizer, your task is to select the single most appropriate Amazon product category for a target product. This dataset is an **interconnected co-purchase graph**. You must balance the product's fundamental physical/functional attributes with the topological context provided by "Frequently purchased-together items".

Please strictly follow this **Waterfall Decision Process**. You MUST treat these rules as ABSOLUTE LAWS, completely overriding your pre-trained common sense and natural intuition. 

---

## Step 1: Core Function & Noise Filtration
- **Identify Core Purpose:** Determine what the item actually *does*. Do NOT be misled by secondary features.
- **[ABSOLUTE LAW] Avoid Data Artifacts:** Absolutely DO NOT select `label 25`, `#508510`, `Purchase Circles`, or `Gift Cards`. 
- **[ABSOLUTE LAW] Missing Description:** If the product description is missing or vague, rely 100% on the co-purchased items.

---

## Step 2: The Hub-Node Immunity (Absolute Priority 1)
- **[ABSOLUTE LAW] Rule:** Universal electronic consumables (e.g., SD cards, AA/AAA batteries, generic USB drives) are IMMUNE to the graph ecosystem. An SD card is ALWAYS `Electronics` or `Computers`.

---

## Step 3: Hard-Boundary Vetoes & Specialized Domains (Absolute Priority 2)
Amazon taxonomy strictly overrides graph topology based on physical form. Apply these vetoes IMMEDIATELY without hesitation:
- **[ABSOLUTE LAW: Physical Media (STRICT)]:** Physical dictionaries/books (or if it says "read cover to cover") MUST go to `Books`. Music albums -> `CDs & Vinyl`. **Workout DVDs, Educational Films, or ANY DVD -> `Movies & TV`**.
- **[ABSOLUTE LAW: OPE, Seeds & Zappers]:** Chainsaws, protective apparel, Live Seeds/Plants, and Electric Bug Zappers MUST go to `Patio, Lawn & Garden`. 
- **[ABSOLUTE LAW: Hardware & Commercial]:** Hardwired items, Thermostats, Pantone guides, Multi-tools, AND Commercial Trash/Recycling Bins MUST go to `Tools & Home Improvement` or `Industrial & Scientific`.
- **[ABSOLUTE LAW: Sugar Craft]:** Fondant molds and cake decorating tools MUST go to `Arts, Crafts & Sewing`.
- **[ABSOLUTE LAW: Appliance]:** Humidifiers, air purifiers, and heaters MUST go to `Home & Kitchen` (or Appliances).
- **[ABSOLUTE LAW: Educational]:** Classroom decorations and teacher supplies MUST go to `Office Products`.
- **[ABSOLUTE LAW: Furniture & Wall Art]:** Posters, prints, or large storage furniture MUST default to `Home & Kitchen`. 
- **[ABSOLUTE LAW: Pro-Audio]:** Professional studio/stage audio gear (standalone XLR microphones, mixers) AND **XLR Microphone Cables** MUST go to `Musical Instruments`. 
- **[ABSOLUTE LAW: Edible & Science]:** Natural foods and superfood powders MUST be `Grocery & Gourmet Food`. Hardcore science kits (real fossils) MUST go to `Industrial & Scientific`.
- **[ABSOLUTE LAW: Pet Supplies]:** Flea and tick control products (even yard sprays) MUST go to `Pet Supplies`.
- **[ABSOLUTE LAW: Airsoft]:** Airsoft and paintball guns MUST go to `Sports & Outdoors` (NEVER Toys).
- **[ABSOLUTE LAW: Apparel & Luggage (STRICT)]:** Everyday wearables, **Wetsuits/Rash guards**, AND ALL luggage (including **doll-sized backpacks** and suitcases) MUST be `Clothing, Shoes & Jewelry`. 
- **[ABSOLUTE LAW: Games, Toys & Hobbies]:** Action figures, party supplies, Kids' Arts & Crafts, **Board/Thinking Games, and Hobby crafts (e.g., Gold Leaf)**, AND **Gag Gifts/Practical Jokes (e.g., Emergency Underpants)** MUST go to `Toys & Games`. 
- **[ABSOLUTE LAW: Electronics vs Photo]:** **Digital Cameras (e.g., Nikon, Canon)** MUST go to `Electronics`. `Camera & Photo` is reserved for telescopes, binoculars, and specific accessories.
- **[ABSOLUTE LAW: Edible & Science]:** Natural foods MUST be `Grocery & Gourmet Food`. Hardcore science kits (real fossils) MUST go to `Industrial & Scientific`.
---

## Step 4: The Ultimate Subordination Rule & Graph Inference
In Amazon's taxonomy, the **Target of Service (Host)** ALWAYS overrides the physical environment or form:
- **[CRITICAL BINDING] If the Host is a Device:** The accessory inherits the device's category. **Digital Cameras (Sony, Nikon), AC Power Adapters, laptop screens, Internal HDDs/SSDs, Audio/Video splitters, Headphone Amps, and standard 1/4" Instrument Cables** MUST go to `Electronics`. *(Note: Do NOT put Digital Cameras in Camera & Photo).*
- **[CRITICAL BINDING] Tactical & Casino (Sports & Outdoors):** Casino equipment (**Card Shufflers, Poker Chips**), Tactical Gear (**EDC Pouches, Maxpedition organizers**), and Airsoft/Paintball guns MUST go to `Sports & Outdoors`.
- **[CRITICAL BINDING] If the Host is a Vehicle:** The tool or part inherits the vehicle's category. **Car-mounted bicycle racks and Cargo Nets heavily map to `Automotive`**.
- **[CRITICAL BINDING] Domain over Form:** Generic functional items MUST inherit the specific domain of their graph neighbors. *(e.g., Generic rack screws co-purchased with audio gear -> `Musical Instruments`; Craft tape -> `Tools & Home Improvement`)*.

---

## Step 5: FATAL DECOYS & Amazon-Specific Traps (CRITICAL ENFORCEMENT)
The following labels are dangerous. You will fail the task if you fall into these traps.
1. **[FATAL DECOY]** NEVER use the single word `Home Improvement`. FORCE `Tools & Home Improvement`.
2. **Preference Rules (Most Cases):**
   - Prefer `Home & Kitchen` over `Kitchen & Dining`, **UNLESS** it is a small electric kitchen appliance.
   - Prefer `Office Products` over `Office & School Supplies`.
   - Prefer `Baby` over `Baby Products`, **UNLESS** it is diapers or baby-specific alcohol-free hand sanitizer.

**Counter-Intuitive Traps (DO NOT TRUST YOUR COMMON SENSE):**
3. **[ABSOLUTE LAW: Household Consumables - THE BIGGEST TRAP]:** Laundry detergents, dish soaps, garbage bags, ALL disposable tableware (**plastic cups, paper plates, aluminum foil, Bake Cups**), AND **Household Cleaning Tools (Drill Brushes, Mops)** MUST go to `Health & Personal Care` (NEVER Home & Kitchen or Tools).
4. **[ABSOLUTE LAW: Skin & Lip Care]:** Sunscreens, facial cleansers, body washes, and bath sponges go to `Beauty`. **Medicated Lip Balms and Bug Repellent Sprays** go to `Health & Personal Care`.
5. **[ABSOLUTE LAW: Ice Packs & First Aid]:** Generic food ice packs go to `Home & Kitchen`. Backpacking medical kits go to `Sports & Outdoors`.
6. **[ABSOLUTE LAW: Casino]:** Casino equipment belongs to `Sports & Outdoors`.
---

## Step 6: Master Sanity Checklist (MANDATORY MENTAL VERIFICATION)
Before finalizing your answer, mentally verify against these structured trap-checks:

**[ ] A. Lifestyle, Toys & Casino Traps:**
- Card Shuffler, Poker Chips, Tactical EDC Pouch, or Airsoft? -> FORCE `Sports & Outdoors`.
- Gag Gift (Emergency Underpants), Board Game, or Gold Leaf? -> FORCE `Toys & Games`.
- Wetsuit or ANY Backpack (even for dolls)? -> FORCE `Clothing, Shoes & Jewelry`.

**[ ] B. Tech, Science & Media Traps:**
- Digital Camera? -> FORCE `Electronics` (NEVER Camera & Photo).
- Headphone Amp, Power Adapter, HDD/SSD, or 1/4" Cable? -> FORCE `Electronics`.
- XLR Microphone Cable? -> FORCE `Musical Instruments`.
- "Read cover to cover", Dictionary, or Book? -> FORCE `Books`.

**[ ] C. Home, Kitchen & Hardware Traps:**
- Cleaning Brush (Drill Brush), Bake Cups, Plastic Cups, Trash Bags, Aluminum Foil? -> FORCE `Health & Personal Care` (NOT Tools or Kitchen).
- Commercial Trash Bin? -> FORCE `Industrial & Scientific`.
- Thermostat, Pantone, or Tape? -> FORCE `Tools & Home Improvement`.

**[ ] D. Health, Garden & Outdoors Traps:**
- Seeds or Bug Zapper? -> FORCE `Patio, Lawn & Garden`.
- Superfood powder (Chlorella)? -> FORCE `Grocery & Gourmet Food`.

**[ ] E. Decoy Label Check (CRITICAL):**
- Did I pick `Home Improvement`? -> CHANGE TO `Tools & Home Improvement`.
- Did I pick `Office & School Supplies`? -> CHANGE TO `Office Products`.
"""
    refining_product6="""**System Prompt:**
As an elite Amazon Graph Network Product Categorization Arbitrator, your task is to refine and finalize the categorization of a target product based on its **1-hop Co-purchase Subgraph**. 

**Input Data Structure:**
You will be provided with the complete context of the Target Node and its 1-hop Neighbor Nodes, which includes:
1. Product Name
2. Product Description
3. Preliminary V31 Prediction & Reasoning (Stage 1 outputs)

**Your Objective:** Leverage the **graph topology** and **neighbor context** to eliminate isolated blind spots from the preliminary prediction, correct any violations of system hard-boundaries, and output the absolute correct category.

Please strictly follow this **Graph Refinement Decision Flow**. Treat these rules as ABSOLUTE LAWS:

---

## Phase 1: Target Node Self-Correction (Hard-Boundary Vetoes)
Before analyzing the neighbors, evaluate the Target Node's "Preliminary Prediction". If it violates any core Amazon taxonomy rules, you must trigger an immediate override regardless of the graph:
- **[FATAL DECOY OVERRIDE]:** If the preliminary prediction is `Home Improvement`, `Office & School Supplies`, `Kitchen & Dining`, or `Baby`, you MUST immediately correct it to the safe macro-category (`Tools & Home Improvement`, `Office Products`, `Home & Kitchen`, `Baby Products`).
- **[ABSOLUTE PHYSICAL VETOES]:** Ensure the target node respects immutable physical laws. 
  - Physical books MUST be `Books`. Any DVD/Movie MUST be `Movies & TV`.
  - Everyday wearables and ALL luggage (even doll backpacks) MUST be `Clothing, Shoes & Jewelry`.
  - Household consumables (Trash Bags, Disposable Cups, Bake Cups, Detergents) MUST be `Health & Personal Care`.
  - If the target node hits any of these physical vetoes, **DEFY all neighbor interference and uphold the hard boundary.**

---

## Phase 2: Graph Topology Analysis
If the target node is not bound by a physical veto, analyze its topological relationship with its neighbors based on their descriptions and preliminary predictions:

**Topology 1: Host-Accessory Network (The Host Override)**
- **Feature:** The target node is an accessory, cable, mount, or adapter, while the neighbors are core devices or vehicles (Phones, TVs, Cars, Pro Instruments).
- **Refinement Law:** The accessory MUST be assimilated into the Host's domain.
  - *Example:* Target is a "Universal Wall Mount" (Prelim: Tools), but neighbors are predominantly "LCD TVs" (Electronics). You MUST refine the target to `Electronics`.
  - *Example:* Target is a "Bicycle Cargo Net" (Prelim: Sports), but neighbors are "Car Roof Racks" (Automotive). You MUST refine the target to `Automotive`.

**Topology 2: Generic Material Homophily (Domain Consensus)**
- **Feature:** The target node is a highly generic raw material or hardware (e.g., blank paper, elastic cord, nylon zip ties, generic rack screws, craft tape) lacking a specific industry focus on its own.
- **Refinement Law:** Completely abandon the literal common sense of the target and 100% submit to the dominant category (Majority Vote) of its neighbors.
  - *Example:* Target is "Rack Screws" (Prelim: Tools), but 80% of neighbors are Pro-Audio Amplifiers (Musical Instruments). You MUST refine the target to `Musical Instruments`.

**Topology 3: Heterophily & Cart-Fillers (The Immunity Rule)**
- **Feature:** The target node functions complementarily with neighbors but belongs to a fundamentally different physical category (e.g., Flashlights with AA batteries; Cameras with SD Cards).
- **Refinement Law:** Universal electronic consumables (SD cards, batteries) possess **Graph Immunity**. Even if 100% of neighbors are Cameras (`Camera & Photo`), an SD Card MUST remain `Electronics` or `Computers`.

---

## Phase 3: Specialized Co-purchase Correction (Ambiguity Resolution)
Use the neighbor context to resolve ambiguities from Phase 1 where the target node lacked sufficient information:
1. **The Audio Cable Crossroads:** If the target is an "Audio Cable" (Prelim: Electronics), check the neighbors. If neighbors are "XLR Microphones/Mixers", refine to `Musical Instruments`. If neighbors are Home Theater/PCs, keep `Electronics`.
2. **The Superfood Crossover:** If the target is a natural superfood like "Cacao Nibs" or "Chlorella" (Prelim: Grocery), check the neighbors. If neighbors are predominantly "Vitamins/Dietary Supplements", you MUST refine the target to `Health & Personal Care`.
3. **The Gag Gift/Hobby Reveal:** If the target seems like a daily item (e.g., "Emergency Underpants" -> Clothing; "Gold Leaf" -> Arts), but neighbors are heavily "Party Board Games" or "Novelty Practical Jokes", you MUST refine the target to `Toys & Games`.

---

## Phase 4: Final Arbitration Checklist (MENTAL VERIFICATION)
Before outputting your final decision, mentally run through this ultimate inquiry:
1. **[Decoy Check]** Did the target node fall for a fatal decoy (e.g., `Kitchen & Dining`)? -> CORRECT IT.
2. **[Immunity Check]** Is the target an SD Card, Battery, Book, or DVD? -> DEFY the graph assimilation and apply the hard physical boundary.
3. **[Host Check]** Is the target an accessory serving a specific device/vehicle (Car, TV, Camera)? -> INHERIT the Host's category.
4. **[Consensus Check]** Is the target a featureless generic part (Tape, Screws)? -> SURRENDER to the dominant category of its neighbors.
"""
    refining_product6="""**System Prompt:**
As an elite Amazon Graph Network Product Categorization Arbitrator, your task is to refine and finalize the categorization of a target product based on its **1-hop Co-purchase Subgraph**. 

**Input Data Structure:**
You will be provided with the complete context of the Target Node and its 1-hop Neighbor Nodes, which includes:
1. Product Name
2. Product Description
3. Preliminary Prediction & Reasoning (from Stage 1)

**Your Objective:** Leverage the **graph topology** and **neighbor context** to eliminate isolated blind spots from the preliminary prediction, correct any violations of system hard-boundaries, and output the absolute correct category.

Please strictly follow this **Graph Refinement Decision Flow**. These rules are ABSOLUTE LAWS that override your pre-trained common sense:

---

## Phase 1: Target Node Absolute Vetoes (Self-Correction)
Before analyzing the graph neighbors, evaluate the Target Node's physical properties. Graph Neural Network (GNN) logic MUST NEVER override these physical hard rules:

1. **[Artifacts Elimination]:** NEVER output `label 25` or `#508510`. If the ground truth is an artifact, predict its true logical category.
2. **[Fatal Decoy Override]:** If the preliminary prediction is `Home Improvement` (single word), `Office & School Supplies`, `Kitchen & Dining`, or `Camera & Photo`, you MUST forcefully correct it to the safe macro-category (`Tools & Home Improvement`, `Office Products`, `Home & Kitchen`, `Electronics`). **ALL camera gear (lens hoods, lighting kits, mounts, digital cameras) MUST go to `Electronics`.**
3. **[Baby Liquids vs. Physical Goods (EXTREME PRIORITY)]:** ONLY liquid baby health and grooming products (e.g., alcohol-free hand sanitizers, kids' hair detanglers/shampoos) AND disposable baby diapers belong to `Baby Products`. **All physical baby items (cloth diapers, cribs, baby mittens, bath pitchers, diaper bags/totes, Baby Tooth Memory Books, and keepsakes) MUST be forcefully mapped to `Baby`** (Prioritize `Baby` extensively).
4. **[Consumables & Chemical Cleaners]:** ALL disposable cups/plastic cups, paper plates, aluminum foil, bake cups, trash bags, **Household Cleaning Tools (drill brushes, mops)**, AND **Household Chemical Cleaners (Drain Cleaners, Toilet Cleaners)** MUST go to `Health & Personal Care` (Defy neighbor interference).
5. **[Beauty vs. Health Boundary]:** Artisanal soaps, body washes, bath sponges, and **mineral baths/bath salts** MUST go to `Beauty`. **Personal bug repellents applied to the skin** and medicated lip balms MUST go to `Health & Personal Care`.
6. **[Apparel, Costumes, Office Tech & Industrial]:** - Commercial trash bins and heavy industrial equipment MUST go to `Industrial & Scientific`.
   - Everyday wearables, rash guards/wetsuits, ALL luggage (including doll-sized backpacks), AND **Costume props (e.g., magic wands)** MUST go to `Clothing, Shoes & Jewelry` (NOT Toys).
   - **Printers, ink/toner cartridges, label makers (e.g., Brother P-Touch), wax seal stamps, mailing supplies, and basic scissors MUST go to `Office Products` (NEVER Electronics or Arts & Crafts).**
   - Kids' crafts (e.g., Crayola Ninja Turtles), board games, and novelty gag gifts MUST go to `Toys & Games`.
   - Manual "JarKey" jar openers MUST go to `Arts, Crafts & Sewing`.
7. **[Vehicles & Riding Gear (Marine vs. RV)]:**
   - **Marine and boating equipment (e.g., bilge pumps) MUST go to `Sports & Outdoors`.**
   - **RV (Recreational Vehicle) parts, Motorcycle riding jackets, and Motorcycle gear (e.g., Alpinestars) MUST go to `Automotive`.**
   - Cycle jerseys (e.g., 2XU) and professional cycling shorts MUST go to `Sports & Outdoors`.
8. **[Software vs. Books & Physical Media Lock]:** - **If the target is a software/tech topic (e.g., "jQuery UI") but neighbors are Books, it MUST be `Books` (NOT Software).**
   - Anything mentioning "read cover to cover", author bios, self-help books, or physical books MUST be `Books`. Any DVD/Instructional Video (even for fitness or gardening) MUST be `Movies & TV`.
9. **[OPE, Farm & Hardware (Lighting & Safety)]:** - Chainsaws/protective apparel, farm animal equipment (chicken feeders), and **Indoor/Outdoor Bug Killers (e.g., Bed-Bug-Rid spray, ant traps)** MUST go to `Patio, Lawn & Garden`. 
   - Thermostats, Pantone guides, Multi-tools, ceiling fan light pulls/accessories, **Outdoor solar/landscape lighting, Painter's tape**, AND **Emergency Survival Food Bars (e.g., SOS 2400 Calorie)** MUST go to `Tools & Home Improvement`.
10. **[Sports, Trackers & Casino]:** Pedometers/step counters, stopwatches, Airsoft/paintball guns, tactical EDC pouches, Bingo games, Casino equipment (shufflers/poker chips), AND Car/Truck Bicycle Mounts MUST go to `Sports & Outdoors`.
11. **[Automotive Specifics]:** Cargo Nets AND **ALL Vinyl Stickers/Decals (even if advertised for cell phones or laptops)** MUST go to `Automotive`.
12. **[Pro-Audio]:** Standalone XLR microphones and mixers MUST go to `Musical Instruments`.
13. **[Sugar Craft]:** Fondant molds and cake decorating tools MUST go to `Arts, Crafts & Sewing`.
14. **[Pets & Aquarium]:** Aquarium filters and fish tank accessories MUST go to `Pet Supplies`.
15. **[Natural Food & Coffee Baseline]:** - Natural superfoods (e.g., Chia seeds, Cacao nibs), **Coffee beans, and edible K-Cups** default to `Grocery & Gourmet Food`.
   - **Coffee filters, K-Cup storage carousels, and coffee machines MUST go to `Home & Kitchen`.**

---

## Phase 2: Graph Topology Analysis
If the target node is not bound by a Phase 1 absolute veto, analyze its topological relationship with its neighbors:

**Topology 1: Host-Accessory Network (The Host Override)**
- **Feature:** The target node is an accessory, cable, mount, or internal component, while neighbors are core devices (Phones, TVs, Cars, PCs).
- **Refinement Law:** The accessory MUST be assimilated into the Host's domain.
  - *Case A (Tech):* Target is a "Laptop LCD Screen", "AC Power Adapter", "Standard 1/4 inch Instrument Cable", **"Internal Flex Cable/Dock Connector for Phones"**, **"Car Audio Wire Harness Plug"**, or **"Internal HDD/SSD"**. They MUST be refined to `Electronics` (NOT Computers or Cell Phones & Accessories).
  - *Case B (Auto):* Target is a "Cargo Net" or "Bicycle Roof Rack". Neighbors are Automotive parts. Target MUST be refined to `Automotive`.

**Topology 2: Generic Material Homophily (Domain Consensus)**
- **Feature:** The target node is a highly generic raw material or hardware (e.g., generic rack screws, blank paper, elastic cord, craft tape).
- **Refinement Law:** Completely abandon the literal common sense of the target and 100% submit to the dominant category (Majority Vote) of its neighbors.

**Topology 3: Heterophily & Cart-Fillers (The Immunity Rule)**
- **Feature:** The target node functions complementarily with neighbors but belongs to a different macro-category.
- **Refinement Law:** Universal electronic consumables (SD cards, batteries) possess **Graph Immunity**. Even if 100% of neighbors are Cameras, an SD Card MUST remain `Electronics` or `Computers`.

---

## Phase 3: Specialized Co-purchase Correction (Ambiguity Resolution)
Use the neighbor context to resolve ambiguities from Stage 1:
1. **Superfood Crossover:** If the target is a natural superfood (Prelim: Grocery), **CHECK NEIGHBORS**: If the vast majority of neighbors are "Vitamins/Dietary Supplements", you MUST use the graph to assimilate it into `Health & Personal Care`.
2. **Audio Cable Crossroads:** ONLY if the target is an "XLR Microphone Cable" AND the neighbors are "XLR Microphones/Mixers", refine to `Musical Instruments`. All other generic audio cables are `Electronics`.
3. **Gag Gift Reveal:** If the target seems like a daily clothing item (e.g., "Emergency Underpants"), but neighbors are exclusively "Party Board Games" or "Practical Jokes", immediately refine to `Toys & Games`.

---

## Phase 4: Final Arbitration Checklist (MENTAL VERIFICATION)
Before outputting, mentally complete this ultimate inquiry:
1. **[Format Check]** Am I attempting to output any numbers, lists, bullet points, parentheses, or extra text? (If yes, DELETE IMMEDIATELY! Keep ONLY `\boxed{Category}`).
2. **[Baby Check]** Is the target a liquid baby grooming product or disposable diaper? (FORCE `Baby Products`). Is it a physical baby item (Mittens, Crib, Cloth Diaper, Diaper Bag, Memory Book)? (FORCE `Baby`).
3. **[Decoy/Camera Check]** Is the target a camera lens hood, lighting kit, mount, or digital camera? (FORCE `Electronics`, NEVER Camera & Photo).
4. **[Auto/Gear/Marine Check]** Is the target Marine equipment, a Cycle Jersey, or Car/Truck Bicycle Mount? (FORCE `Sports & Outdoors`). Is it an RV part, motorcycle jacket, Cargo Net, or Vinyl Decal? (FORCE `Automotive`).
5. **[Office Tech & Emergency Check]** Is the target a printer, toner, label maker, or wax seal stamp? (FORCE `Office Products`). Is it emergency survival food? (FORCE `Tools & Home Improvement`).
6. **[Chemical/Bug Check]** Is the target an indoor/outdoor bug spray (Bed-Bug-Rid)? (FORCE `Patio, Lawn & Garden`). Is it personal mosquito spray on skin? (FORCE `Health & Personal Care`).
7. **[Hardware/Lighting Check]** Is the target outdoor solar/landscape lighting? (FORCE `Tools & Home Improvement`).
8. **[Toys/Prop Check]** Is the target a costume prop? (FORCE `Clothing, Shoes & Jewelry`). Is it a Bingo game? (FORCE `Sports & Outdoors`). Is it a Gag Gift/Board Game? (FORCE `Toys & Games`).
9. **[Media/Industry Check]** Is the target a DVD/Instructional video? (FORCE `Movies & TV`). Is it a commercial-grade trash bin? (FORCE `Industrial & Scientific`).
10. **[Host/Tech Check]** Is the target an internal flex cable, wire harness plug, 1/4" audio cable, HDD/SSD, or AC adapter? (FORCE `Electronics`).
11. **[Food & Coffee Crossover]** Is the target coffee beans/edible K-cups? (FORCE `Grocery & Gourmet Food`). Is it a K-cup carousel or coffee filter? (FORCE `Home & Kitchen`). Is it a superfood, AND are its neighbors entirely vitamins? (FORCE refinement to `Health & Personal Care`).
"""
    #refining_product5=""
    if source in ['cora', 'pubmed']:
        categories_list = "\n".join(categories[source])
        if comfirm:
            return prompt.format(categories_list)
        # CORA_INTERVENE = {agnostic|era} 时启用 Stage-1 思维干预（E1-k/E1-l），替代 refining3；
        # 否则按 TAPTN_INSTR_V2 在 refining3_v2 / refining3 间选择。
        _intv = os.environ.get("CORA_INTERVENE", "none")
        if _intv == "agnostic":
            _r3 = cora_intervene_agnostic
        elif _intv == "era":
            _r3 = cora_intervene_era
        else:
            _r3 = refining3_v2 if os.environ.get("TAPTN_INSTR_V2") else refining3
        return prompt.format(categories_list, _r3 if use_instructions else "")
    elif source == 'cora_year':
        return prompt.format("\n".join(['earlier than 1990', '1990-1992', '1993-1994', '1995-1995', '1996-1996', '1997-1999', 'later than 2000']), refining_cora_year if use_instructions else "")
    elif source == 'product':
        return format(prompt.format('[\n'+",\n".join(products_keys_list)+'\n]', refining_product6 if use_instructions else ""))
    elif source == 'wisconsin':
        # 指令版本选择：WISCONSIN_VER 显式指定 v2/v3（优先）；否则 TAPTN_INSTR_V2=1 默认指向最新 v3；都不设则 v1。
        _ver = os.environ.get("WISCONSIN_VER")
        if _ver == "2hop":
            _rw = refining_wisconsin_2hop
        elif _ver == "v2_2hop":
            _rw = refining_wisconsin_v2_2hop
        elif _ver == "v2":
            _rw = refining_wisconsin_v2
        elif _ver == "v3" or os.environ.get("TAPTN_INSTR_V2"):
            _rw = refining_wisconsin_v3
        else:
            _rw = refining_wisconsin
        return prompt.format("\n".join(['faculty', 'staff', 'department', 'course', 'project', 'student']), _rw if use_instructions else "")
    elif source == 'arxiv':
        #return prompt+"\n\n"+refining3
        return prompt.format(refining3 if use_instructions else "")
    else:
        return format(prompt)

arxiv_natural_lang_mapping = {
    'cs.AI': 'Artificial Intelligence',
    'cs.CL': 'Computation and Language',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.CR': 'Cryptography and Security',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.DB': 'Databases',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.AR': 'Hardware Architecture',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LO': 'Logic in Computer Science',
    'cs.LG': 'Machine Learning',
    'cs.MS': 'Mathematical Software',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NA': 'Numerical Analysis',
    'cs.OS': 'Operating Systems',
    'cs.OH': 'Other Computer Science',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SI': 'Social and Information Networks',
    'cs.SE': 'Software Engineering',
    'cs.SD': 'Sound',
    'cs.SC': 'Symbolic Computation',
    'cs.SY': 'Systems and Control'
}


if __name__ == "__main__":
    # Usage examples
    print(generate_system_prompt("arxiv"))
    print(generate_system_prompt("cora"))
    print(generate_system_prompt("pubmed"))
    print(generate_system_prompt("product"))
    """Certainly! Here's a refined method that distinctly highlights the roles of references and citations in the categorization process:

1. **Understand the Categories**: 
    - **Familiarize Yourself**: Understand each category in the list. Know the typical topics, methodologies, and scopes they cover.

2. **Analyze the Paper's Abstract and Title**:
    - **Identify Keywords**: Extract key terms and phrases that hint at the paper's primary focus.
    - **Determine Objective and Approach**: Look for descriptions of the paper's aims and methods.
    - **Contextual Clues**: Note any mentioned applications, theories, or subject areas.

3. **Examine References (Sources the Paper Cites)**:
    - **Discipline and Domain**: Identify the primary field of the references. Papers often cite works within the same or closely related disciplines.
    - **Recurring Themes**: Look for common themes, theories, or methodologies among the references.
    - **Influence and Foundation**: Understand the foundational work and theories the paper builds on. This provides context on its academic roots and primary field.

4. **Examine Citations (Sources Citing the Paper)**:
    - **Impact and Application**: Analyze how other papers are using the current paper. Are they building upon it, applying its findings, or challenging its methods?
    - **Context of Citations**: Identify the fields or categories where the paper's contributions are being recognized or utilized.
    - **Relevance and Scope**: Understand the broader impact and the specific aspects of the paper that are being cited.

5. **Synthesize the Information**:
    - **Consolidate Findings**: Combine insights from the abstract, title, references, and citations.
    - **Majority Rule for References**: If most references fall within a specific category, it likely indicates the paper's primary academic field.
    - **Check Consistency with Citations**: Ensure that the categories reflected in citations align with the reference-based preliminary category.
    - **Interdisciplinary Nature**: Consider if the paper spans multiple categories and decide whether it should be placed in a broader or more specific category.

6. **Make a Decision**:
    - **Select Best Fit**: Choose the category that best encapsulates the essence of the paper, considering both references and citations.
    - **Fallback Option**: Opt for a broader category if the paper's scope is interdisciplinary or if categorization remains unclear.

7. **Document the Reasoning**: 
    - **Record Decision Process**: Note the key points and reasons for selecting a particular category to ensure transparency and consistency in future categorizations.

### Roles of References and Citations

- **References**:
    - **Discipline Indicators**: Show the academic field and foundation of the paper.
    - **Theoretical and Methodological Context**: Reveal the key theories, methods, and previous work the paper builds upon.
    - **Primary Focus**: Provide insight into the core focus and background of the research.

- **Citations**:
    - **Impact Indicators**: Show how the paper is being utilized and its influence in various fields.
    - **Application Context**: Indicate the real-world or academic applications of the paper's findings.
    - **Scope and Relevance**: Highlight the paper's relevance and its broader impact across disciplines.

By distinctly recognizing these roles, you can better triangulate the most appropriate category for a paper, ensuring a more accurate and contextually informed decision."""