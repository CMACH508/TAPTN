import sys
sys.path.append("./")

refining3_2="""
### Cora Paper Classification Protocol

The Cora dataset contains machine learning papers from the late 1990s and early 2000s, classified into exactly **7 categories**. Your task is to identify which category a paper belongs to based on its title, abstract, and its citation/reference network.

---

#### Category Definitions & Keyword Signals

Use the following definitions and keyword signals to identify each category:

1. **Rule Learning**
   - *What it covers*: Learning explicit, interpretable IF-THEN rules or logical clauses from data. Primarily symbolic/logic-based methods.
   - *Keywords*: inductive logic programming (ILP), rule induction, FOIL, RIPPER, Prolog, Horn clauses, propositional rules, decision rules, first-order logic, concept learning, rule extraction.
   - *Distinguished from Case Based*: Rule Learning produces *general* rules; Case Based stores *specific* past examples.

2. **Neural Networks**
   - *What it covers*: Architectures and training algorithms for artificial neural networks.
   - *Keywords*: neural network, backpropagation, perceptron, feedforward, recurrent network, LSTM, radial basis function, activation function, gradient descent, connectionist, multi-layer.
   - *Distinguished from Probabilistic Methods*: Neural Networks focus on network structure and weights; Probabilistic Methods focus on probability distributions and Bayesian inference.

3. **Case Based**
   - *What it covers*: Case-Based Reasoning (CBR) — solving new problems by retrieving and adapting solutions from stored past cases.
   - *Keywords*: case-based reasoning, CBR, analogical reasoning, case retrieval, nearest neighbor, similarity measure, case adaptation, memory-based learning, instance-based learning.
   - *Distinguished from Rule Learning*: Case Based keeps specific cases; Rule Learning extracts generalized rules.

4. **Genetic Algorithms**
   - *What it covers*: Evolutionary and population-based optimization algorithms inspired by natural selection.
   - *Keywords*: genetic algorithm, evolutionary computation, crossover, mutation, fitness function, genetic programming, evolutionary strategy, population-based, selection pressure, chromosomes.
   - *Distinguished from Reinforcement Learning*: Genetic Algorithms optimize a population without agent-environment interaction; Reinforcement Learning trains a single agent through sequential decisions and rewards.

5. **Computational Learning Theory**
   - *What it covers*: Formal mathematical analysis of learning algorithms — their computational complexity, sample efficiency, and theoretical limits.
   - *Keywords*: PAC learning, VC dimension, sample complexity, learnability, probably approximately correct, Vapnik-Chervonenkis, computational complexity of learning, mistake bounds, boosting theory.
   - *Distinguished from all others*: This is a **theoretical** category. Papers prove theorems and derive bounds; they do NOT primarily report empirical experiments on real datasets.

6. **Reinforcement Learning**
   - *What it covers*: Learning through sequential interaction with an environment, guided by reward signals.
   - *Keywords*: reinforcement learning, reward, policy, value function, Q-learning, Markov decision process (MDP), temporal difference (TD), actor-critic, exploration vs exploitation, agent, environment.
   - *Distinguished from Genetic Algorithms*: Reinforcement Learning uses a single agent and reward feedback; Genetic Algorithms use population evolution.

7. **Probabilistic Methods**
   - *What it covers*: Probabilistic and Bayesian approaches to machine learning and reasoning under uncertainty.
   - *Keywords*: Bayesian network, probabilistic graphical model, Markov random field, hidden Markov model (HMM), expectation-maximization (EM), belief propagation, prior/posterior, naive Bayes, uncertainty, probabilistic inference.
   - *Distinguished from Neural Networks*: Probabilistic Methods are centered on probability distributions; Neural Networks are centered on weight optimization in network architectures.

---

#### Step-by-Step Decision Protocol

**Step 1: Screen for Computational Learning Theory**

*First*, check if the paper is **primarily theoretical**.
- Does the abstract focus on proofs, formal analysis, complexity bounds, or learnability guarantees?
- Is there little or no empirical evaluation on real-world datasets?
- Do the references cite foundational theory papers (Valiant, Vapnik, Blumer, Kearns)?

➡ If YES → **Computational Learning Theory**. Stop here.

---

**Step 2: Check for Domain-Exclusive Signals**

Look for strong category-exclusive keywords in the **paper's own title and abstract**:

| Strong Signal | Category |
|---|---|
| "genetic algorithm", "crossover", "mutation", "fitness", "evolutionary" | **Genetic Algorithms** |
| "reinforcement learning", "Q-learning", "MDP", "reward", "policy" | **Reinforcement Learning** |
| "case-based reasoning", "CBR", "case retrieval", "case adaptation" | **Case Based** |
| "inductive logic programming", "ILP", "rule induction", "FOIL", "Horn clause" | **Rule Learning** |
| "Bayesian network", "graphical model", "HMM", "EM algorithm", "prior probability" | **Probabilistic Methods** |
| "neural network", "backpropagation", "perceptron", "connectionist" | **Neural Networks** |

➡ If a **strong exclusive keyword** is found → Assign that category. Stop here.

---

**Step 3: Use the Citation/Reference Network as Evidence**

If the paper's own text is ambiguous, examine its **neighbors** (references and citations):
- Tally how many cited/citing papers belong to each category.
- A **majority (≥60%)** in one category strongly suggests the paper belongs there.
- Pay attention to the **category of the citing paper's topic**, not just the authors.

---

**Step 4: Resolve Ambiguous Cases with Tie-Breaking Rules**

| Conflict | Resolution Rule |
|---|---|
| Neural Networks vs. Probabilistic Methods | If the paper proposes a **network architecture** or training algorithm → Neural Networks. If it reasons about **probability distributions** → Probabilistic Methods. |
| Rule Learning vs. Case Based | If the output is a **set of rules/clauses** → Rule Learning. If it **stores and retrieves examples** → Case Based. |
| Genetic Algorithms vs. Reinforcement Learning | If there is a **population of solutions** evolving → Genetic Algorithms. If there is a **single agent** interacting with an environment → Reinforcement Learning. |
| Computational Learning Theory vs. Neural Networks | If the paper **proves theoretical properties** of neural networks (VC dimension, convergence bounds) → Computational Learning Theory. If it **proposes or trains** a network → Neural Networks. |

---

**Step 5: Final Decision and Self-Check**

- Select the **single best-fit category** from the 7 options.
- Do NOT default to "Neural Networks" as a fallback — all 7 categories are equally valid choices.
- Verify: Does the chosen category align with (a) the paper's own abstract, AND (b) the majority pattern in its citation network?
- If these two signals conflict, **prioritize the paper's own content** over the network.
- Perform a final review for any inconsistencies and correct them before stating the final category.

"""

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

refining5_2="""
### Cora Second-Iteration Revision Protocol

You are now in the **second iteration** of classification. You have already produced an initial categorization for the target paper and for each of its neighbors. Your task is to **critically review and refine** that initial judgment using the richer context now available.

The 7 valid categories are:
- **Rule Learning** — Learning explicit IF-THEN rules or logic clauses (keywords: ILP, rule induction, FOIL, Horn clauses, propositional rules)
- **Neural Networks** — Neural architectures and training algorithms (keywords: backpropagation, perceptron, recurrent network, feedforward, connectionist)
- **Case Based** — Case-Based Reasoning, solving problems via stored past examples (keywords: CBR, case retrieval, similarity measure, analogical reasoning, instance-based)
- **Genetic Algorithms** — Population-based evolutionary optimization (keywords: genetic algorithm, crossover, mutation, fitness, evolutionary computation)
- **Computational Learning Theory** — Formal theoretical analysis of learning (keywords: PAC learning, VC dimension, sample complexity, learnability, boosting theory)
- **Reinforcement Learning** — Agent learning via reward signals (keywords: Q-learning, MDP, policy, value function, temporal difference, agent, environment)
- **Probabilistic Methods** — Bayesian and probabilistic reasoning (keywords: Bayesian network, HMM, EM algorithm, graphical model, prior/posterior, belief propagation)

---

#### Phase 1: Audit the Initial Categorization

Start by critically evaluating the initial categorization of the **target paper**:

1. **Re-read the initial reasoning**. Identify whether it was based on:
   - ✅ **Strong signals** (explicit keywords, clear methodological match) → initial result is likely correct.
   - ⚠️ **Weak signals** (vague thematic similarity, genre of research field) → initial result may need revision.
   - ❌ **Known confusion patterns** (see Phase 3) → high revision priority.

2. **Check for reasoning errors**:
   - Did the reasoning confuse the *application domain* (e.g., a machine learning paper applied to robotics) with the *methodology* (e.g., the actual method used is Reinforcement Learning)?
   - Did the reasoning over-rely on a single neighbor's category rather than the paper's own content?

---

#### Phase 2: Evaluate Neighbor Evidence

Use the initial classifications of all neighbor papers (references + citations) as a validation signal:

1. **Tally neighbor categories**:
   - Count how many neighbors belong to each of the 7 categories.
   - Compute the **dominant neighbor category** (highest count).

2. **Apply the evidence thresholds**:

   | Neighbor Consensus | Action |
   |---|---|
   | ≥70% of neighbors in one category, and it **matches** the initial result | **Strong confirmation** — maintain initial categorization. |
   | ≥70% of neighbors in one category, and it **contradicts** the initial result | **Strong challenge** — seriously consider revising. |
   | 50–69% majority, contradicting initial result | **Moderate challenge** — revise only if the paper's own content also has weak alignment to the initial category. |
   | No clear majority (<50% in any one category) | Neighbor network is heterogeneous — **do not revise based on network alone**; rely on the paper's own content. |

3. **Quality-weight the neighbor evidence**:
   - Neighbors whose initial reasoning was strong and well-supported carry more weight than those with weak or uncertain initial categorizations.
   - Ignore neighbors classified as a *fallback* or with explicit uncertainty in their reasoning.

---

#### Phase 3: Check for Common First-Pass Errors

The first iteration is most likely to have made mistakes in these specific confusion pairs. For each, apply the correction test:

| Suspect Confusion | Correction Test |
|---|---|
| **Rule Learning ↔ Case Based** | Does the paper output *general rules* (→ Rule Learning) or *retrieve specific stored examples* (→ Case Based)? |
| **Genetic Algorithms ↔ Reinforcement Learning** | Is there a *population of solutions* evolving (→ GA) or a *single agent* receiving rewards (→ RL)? |
| **Neural Networks ↔ Computational Learning Theory** | Does the paper *prove theoretical bounds* about networks (→ Theory) or *propose/train* a network (→ Neural Networks)? |
| **Neural Networks ↔ Probabilistic Methods** | Is the focus on *network weight optimization* (→ Neural Networks) or *probability distributions and Bayesian inference* (→ Probabilistic Methods)? |
| **Any category ↔ Computational Learning Theory** | If the paper has **no empirical experiments** and focuses on **proofs and formal guarantees**, it almost certainly belongs to **Computational Learning Theory** regardless of topic domain. |

---

#### Phase 4: Decision — Maintain or Revise

Apply this decision rule:

- **MAINTAIN** the initial categorization if:
  - The initial reasoning was based on strong, explicit signals, AND
  - Neighbor evidence does not show a strong majority (≥70%) contradicting it.

- **REVISE** the initial categorization if:
  - The initial reasoning was weak or based on a known confusion pattern (Phase 3), OR
  - Neighbor consensus is ≥70% in a different category AND the paper's own content is at least weakly compatible with that category.

- **NEVER revise** solely because of a minority of cross-category neighbors (<30%). Cross-category citations are normal in machine learning research.

- **NEVER introduce a second category**. This is a single-label classification task; select exactly one category.

---

#### Phase 5: Final Self-Check

Before stating the final category:
1. Confirm the chosen category has at least one direct keyword match in the paper's own title or abstract.
2. Confirm the chosen category is consistent with ≥50% of the neighbors (or explain why the paper's own content overrides the network).
3. If you are changing the initial category, state explicitly: "Revised from [initial] to [new] because..."
4. Verify your final answer is one of the 7 valid categories and is stated clearly.

"""
refining3_3="""By examining a paper's "first-order neighborhood" (its references and citations), you are combining semantic text with structural community context. To avoid common LLM misclassifications on the historical Cora dataset, you MUST strictly follow this step-by-step protocol.



#### Step 1: The Anti-Artifact "Confrontation" Mechanism (CRITICAL DEFENSE)

The Cora dataset contains severe PDF parsing errors where the Abstract may belong to a completely different paper or be garbled text.

- **Action**: You MUST explicitly confront the Title vs. the Abstract. 

- **Rule**: If there is a massive conceptual mismatch (e.g., Title is about "Dynamical Systems" but Abstract is about "Case-based design"), or the text is a bibliography string, this is an ARTIFACT. YOU MUST IGNORE THE ABSTRACT and rely exclusively on the Title and Citation Network.



#### Step 2: Calibrated Category Definitions & Rigid Overrides (Cora-Specific)

Do not use your modern, generic understanding of machine learning. Adhere to these rigid historical boundaries:



- **Neural Networks (The Historical "Black Hole")**: Acts as a massive umbrella for NIPS-era domains.

  - *Biological Override*: ANY paper predicting "Protein Structure", "DNA Sequence", "Finding Genes", or "Sequence Assembly" belongs here, EVEN IF it uses Decision Trees or HMMs.

  - *Theoretical Exemption*: Papers calculating "VC dimension", "PAC bounds", or "sample complexity" SPECIFICALLY and EXCLUSIVELY for Neural Networks/Perceptrons belong HERE, not in Theory.

  - *Methodological Override*: Includes "Gaussian Processes", "Bayesian Non-linear Modelling", "Dynamical Systems", and "Mixtures of Experts" (do NOT place in Probabilistic Methods).



- **Computational Learning Theory (Theory)**: The mathematical limits of learning.

  - *Trump Cards*: "Active Learning", "Query by Committee", "Boosting", "AdaBoost", "Error-Correcting Output Codes (ECOC)", "PAC learning", "Mistake bounds", "finite automata (DFA) learning".

  - *Perspective Rule*: Mathematical bounds for Bayesian/Probabilistic models (e.g., PAC-Bayes) belong here.



- **Case Based**: "Lazy Learning", "Instance-based learning", k-NN.

  - *Strong Indicator*: Papers focusing purely on "Feature Selection", "Feature Weighting", or "Attribute Selection" algorithms heavily lean here (as they solve the curse of dimensionality for k-NN), UNLESS explicitly applied to Decision Trees.



- **Rule Learning**: Extracting human-readable logic. Includes "Decision Trees" (unless applied to DNA), "Theory Refinement" (refining domain theories like KBANN), "Inductive Logic Programming (ILP)", and "Constructive Induction".



- **Probabilistic Methods**: Applied statistical inference, Bayesian belief networks, HMMs, and EM algorithms (strictly OUTSIDE the context of continuous Neural architectures or Computational Biology).



- **Reinforcement Learning**: MDPs, Q-Learning, delayed rewards, policy iteration, dynamic programming for control.



- **Genetic Algorithms**: Evolutionary computation, mutation, fitness, simulated breeding.



#### Step 3: Object vs. Tool vs. Perspective Analysis

Classify based on the **Primary Object of Study** or the **Final Domain Goal**, not the localized tool.

- *Example A*: If ECOC (Theory Domain) uses Decision Trees (Rule Domain) as a testbed -> **Computational Learning Theory**.

- *Example B*: If a Genetic Algorithm (Tool) is used to prune a Decision Tree (Rule Domain) -> **Rule Learning**.

- *Example C*: If a Probabilistic Classifier (Tool) is used for Uncertainty Sampling (Theory Domain) -> **Computational Learning Theory**.



#### Step 4: The "Community Override" Protocol

The citation network (incoming and outgoing edges) reflects the true scientific community. If a paper's text is ambiguous, misleading, or heavily focuses on a specific tool, let the citation network dictate the category. If citations heavily feature "Markov Decision Processes", predict Reinforcement Learning regardless of the paper's own buzzwords.



#### Step 5: Structured Explicit Reasoning

Before providing your final answer, explicitly write out your thought process:

1. **Artifact Check**: Do Title and Abstract align? (If NO -> state that you are Ignoring the Abstract).

2. **Object/Perspective Analysis**: What is the core object being studied? Distinguish the method/tool from the ultimate theoretical/practical goal.

3. **Domain/Trump Card Check**: Are there DNA/Protein keywords? Are there Boosting/Active Learning keywords? (Apply Step 2 Overrides).

4. **Neighborhood Consensus Analysis**: Which specific community dominates the references and citations? Does it contradict or support the text?



#### Step 6: Final Prediction

Based on the synthesis above, predict the top 2 most appropriate categories. For each category you predict, give a relevance score between 0 and 1. Output the final prediction clearly."""
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

refining_actor="""Choosing the most appropriate category for an actor based on their Wikipedia information and collaboration network involves analyzing their career patterns, geographic origins, and professional collaborations. Here is a structured approach to help you categorize an actor:

1. **Understand the Categories**: First, familiarize yourself with the five predefined categories:
   - "American film actors (only)": Actors exclusively working in American films
   - "American film actors and American television actors": Actors working in both American film and television
   - "American television actors and American stage actors": Actors working in American television and stage
   - "English actors": Actors from England working in film, television, or stage
   - "Canadian actors": Actors from Canada

2. **Analyze the Actor's Wikipedia Information**:
   - **Geographic Origin**: Identify nationality and birthplace. Look for clear indicators like "American", "English", or "Canadian".
   - **Career Focus**: Determine primary work areas (film, television, stage) from the description.
   - **Career Scope**: Note if the actor works exclusively in one medium or multiple mediums.
   - **Notable Works**: Identify representative works that indicate their primary career focus.

3. **Examine the Collaboration Network**:
   - **Collaborator Categories**: Review the categories of actors who appear on the same Wikipedia page (collaborators).
   - **Collaboration Patterns**: Note recurring patterns - if most collaborators belong to a specific category, this indicates similar career paths.
   - **Collaboration Count**: Higher collaboration counts with actors in a specific category suggest stronger alignment with that category.

4. **Category-Specific Indicators**:
   - **"American film actors (only)"**: Primarily American nationality, exclusively film work, no significant television or stage presence.
   - **"American film actors and American television actors"**: American nationality, active in both film and television industries.
   - **"American television actors and American stage actors"**: American nationality, active in television and stage theater.
   - **"English actors"**: English nationality or British origin, may work across film, television, or stage.
   - **"Canadian actors"**: Canadian nationality, may work in various mediums.

5. **Synthesize the Information**:
   - **Consistency Check**: Ensure the geographic origin matches the category (e.g., English nationality for "English actors").
   - **Career Pattern Match**: Verify that the actor's work history aligns with the category's medium focus.
   - **Collaboration Alignment**: Check if collaborators share similar categories, indicating similar career trajectories.
   - **Majority Rule**: If most information points to one category, that's likely the correct classification.

6. **Handle Ambiguity**:
   - **Multiple Mediums**: If an actor works in multiple mediums, choose the category that best captures their combined work (e.g., film + television).
   - **Geographic Priority**: When nationality is clear but career medium is ambiguous, prioritize geographic categories (English/Canadian) over medium-specific ones.
   - **Primary vs. Secondary Work**: Focus on the actor's primary and most prominent career area rather than occasional work.

7. **Make a Decision**:
   - **Best Fit**: Choose the category that best represents the actor's nationality and primary career focus.
   - **Single Category**: Select only one category that most accurately captures the actor's profile.

8. **Final Review**:
   - Review the reasoning process and final categorization carefully.
   - Identify any factual errors, inconsistencies, or missing important information.
   - Verify that the category logically fits with the actor's Wikipedia information and collaboration network.
   - Ensure the decision reflects both geographic origin and career pattern accurately.

This method relies on careful analysis of geographic indicators, career patterns, and professional networks. It requires attention to detail and the ability to synthesize information from both individual profiles and collaboration patterns.
"""

refining_actor="""This guide is designed to help you manually classify actor nodes in the graph. Since the explicit category labels (like "American film actor") have been removed from the text to prevent cheating (label leakage), you must act like a detective. You will rely on **context clues** (keywords in the bio) and **social circles** (who they hang out with/collaborate with) to determine the correct label.

This is a **Heterophilic Graph Classification** task, but it relies heavily on **Homophily** (birds of a feather flock together).

### The 5 Target Categories

Your goal is to bin every actor into exactly one of these:

1. **English actors** (Film, TV, or Stage)
2. **Canadian actors** (Any type)
3. **American television and stage actors** (No Film label here)
4. **American film and television actors** (The "Multimedia Star")
5. **American film actors (only)** (The "Movie Star" purist)

---

### Step-by-Step Classification Protocol

Follow this 3-step decision tree for every node.

#### Step 1: The Nationality Filter (The "Passport" Check)

First, check if the actor is non-American. The "English" and "Canadian" categories take precedence over the detailed American genre splits.

* **Scan the Node Text & Neighbors for:**
* **Keywords:** "British", "English", "London", "UK", "Royal", "Oxfordshire", "Canadian", "Toronto", "Vancouver", "Montreal", "Commonwealth".
* **Neighbor Clues:** If the collaborators are predominantly "People from London" or have "British" awards (e.g., "OBE", "Knighthood"), the target is likely **English**.


* **Decision:**
* If signals point to UK/England  **Label: English actors**
* If signals point to Canada  **Label: Canadian actors**
* If signals point to USA (e.g., "American", "New York", "California", "Hollywood")  **Go to Step 2**.



#### Step 2: The "Stage" Check (The Theater Actor)

If the actor is American, check if they belong to the specific "TV + Stage" niche. Note that the label is *specifically* "American television actors and American stage actors".

* **Scan Text for:**
* **Keywords:** "Broadway", "Theatre", "Tony Award", "Playwright", "Stage", "Musical", "Drama Desk Award".


* **Neighbor Clues:** Do they collaborate with other people labeled "Broadway actors" or "Shakespearean actors"?
* **Decision:**
* If you see strong "Stage/Theater" keywords combined with TV keywords  **Label: American television actors and American stage actors**
* If no Stage keywords  **Go to Step 3**.



#### Step 3: The Film vs. TV Split (The Hardest Part)

You are now left with two American choices: **"Film (only)"** vs. **"Film + TV"**.

* **"Film Only"** implies the actor is a pure movie star and does not have the "Television actor" category tag.
* **"Film + TV"** implies they do both.
* **Scan Text for TV Indicators:**
* **Keywords:** "Television", "Series", "Sitcom", "Soap Opera", "Game show", "Talk show host", "Emmy Award" (Daytime or Primetime), "SNL", "Reality".


* **Use the Neighbors (Crucial here):**
* Look at the collaborators. Are they famous for TV? (e.g., "American television talk show hosts", "Game show hosts").
* If the neighbors are heavily associated with specific TV networks or formats, the target is likely in the **Film + TV** bucket.


* **Decision:**
* If you find explicit TV clues or strong TV social circles  **Label: American film actors and American television actors**
* If the text is purely "Academy Award", "Blockbuster", "Film star" with *zero* mention of television/series  **Label: American film actors (only)**



---

### Walkthrough Examples (Applying the Logic)

Let's apply this guide to the data you provided.

#### Case 1: Stanley Holloway

* **Step 1 (Nationality):**
* *Node Clues:* "British Army", "Royal Irish", "People from East Ham" (London).
* *Neighbor Clues:* Sylvia Syms ("People from London"), Ralph Richardson ("British knighthoods").
* *Analysis:* Strong British signals.


* **Result:** **English actors**

#### Case 2: Kenan Thompson

* **Step 1 (Nationality):**
* *Node Clues:* "African American actors", "Georgia (U.S. state) actors". -> **American**.


* **Step 2 (Stage):**
* *Node Clues:* No mention of Broadway, Tony, or Theater. -> **Skip**.


* **Step 3 (Film vs. TV):**
* *Node Clues:* "American child actors". (Ambiguous, could be film or TV).
* *Neighbor Clues:*
* **Bill Hader:** Famous SNL alum (TV).
* **Whoopi Goldberg:** "American television talk show hosts", "Daytime Emmy".
* **Amy Poehler:** "American comedians" (TV heavy).


* *Analysis:* His social circle is dominated by TV comedians and talk show hosts. Even though his raw text doesn't scream "TV", his network does.


* **Result:** **American film actors and American television actors**

#### Case 3: Ted Bessell

* **Step 1 (Nationality):**
* *Node Clues:* "New York actors", "People from Queens". -> **American**.


* **Step 2 (Stage):**
* *Node Clues:* No Stage keywords found. -> **Skip**.


* **Step 3 (Film vs. TV):**
* *Neighbor Clues:*
* **Jackie Gleason:** "American game show hosts", "American comedians". (Strong TV).
* **Doris Day:** "American television talk show hosts". (Strong TV).
* **Rock Hudson:** "Gay actors" (Hudson was famous for both Film and TV).


* *Analysis:* He is collaborating with Game Show hosts and Talk Show hosts. This is a strong indicator of the "Television" category.


* **Result:** **American film actors and American television actors**

---

### Summary Cheat Sheet

| If you see... | And neighbors are... | The Label is... |
| --- | --- | --- |
| **UK / London / British** | British / Knights / Royal | **English actors** |
| **Canadian / Toronto** | Canadian | **Canadian actors** |
| **Broadway / Tony / Theater** | Stage actors / Playwrights | **American TV & Stage** |
| **Talk Show / Emmy / Series** | Game show hosts / TV stars | **American Film & TV** |
| **Oscars / Film (No TV)** | Movie stars (No TV hosts) | **American Film (only)** |
"""

refining_actor="""
### Classification Protocol

#### Step 1: The Strict Nationality Filter (The "Passport" Check)

**Priority: High.** If an actor fits here, **STOP**. Do not check American categories.

* **Rule A: Geography Overrides Neighbors.**
* Ignore who they collaborate with. Look *only* at the **Node Information** first.
* If the text says "People from [City], **Ontario / British Columbia / Quebec / Alberta / Manitoba**", classify as **Canadian**.
* If the text says "People from **London / UK / England / Scotland / Wales**", classify as **English**.


* **Rule B: The "Awards & Titles" Check.**
* **Canadian Signals:** "Genie Award" (Canadian Oscar), "Juno Award", "First Nations" (distinct from Native American), "Order of Canada".
* **English Signals:** "OBE", "CBE", "Knighthood", "Dame", "Royal Shakespeare Company", "Alumni of RADA/Central School of Speech".


* **Rule C: Beware the Ancestry Trap.**
* "Irish-American", "French Canadian American", or "Italian-American" means they are **American**. Do not classify them as English or Canadian based on ancestry alone.



> **Correction form Error Cases:** In Case 17, the model missed "People from Brampton, Ontario". In Case 16, it missed "First Nations". These are immediate triggers for **Canadian**.

---

#### Step 2: The "Stage & TV" Niche (The Theater Professional)

If Step 1 confirms they are American, check this category *before* the general Film/TV split.

* **Target Label:** `American television actors and American stage actors`
* **The Profile:** These are actors known for Broadway, Musicals, and Soap Operas/TV series, but *lacking* a massive Blockbuster Film presence.
* **Keywords to Watch:**
* "Broadway", "Theatre", "Tony Award", "Drama Desk Award", "Musical theatre".
* **AND** TV indicators: "Soap opera", "Daytime Emmy", "Series regular".


* **Negative Constraint:** If they are listed as "Academy Award winner" (Oscars) for *Acting*, they likely belong in a Film category, not here.

> **Correction from Error Cases:** Case 9 (Elizabeth Ashley) had "Tony Award winners". This keyword is a primary driver for this category.

---

#### Step 3: The "Film Only" vs. "Film + Television" Split

This is where the most errors occurred (over 100 cases). You must distinguish between a "Movie Star" and a "Working Actor".

**A. The "Film + Television" Actor (The Generalist)**
Classify as `American film actors and American television actors` if *any* of the following are true:

1. **Voice Acting:** The text contains "**American voice actors**". (Voice work is predominantly TV cartoons/video games).
2. **TV Formats:** Text contains "Sitcom", "Soap opera", "Game show", "Talk show", "Reality", "SNL", "Saturday Night Live".
3. **TV Awards:** "Emmy Award" (Primetime or Daytime), "Golden Globe for **Television**".
4. **The Social Circle (Tie-Breaker):** If the node text is ambiguous, look at the **Collaborators**. If they hang out with "American game show hosts", "Stand-up comedians" (who usually have TV sitcomes), or "TV Anchors", they are in this category.

**B. The "Film Only" Actor (The Purist)**
Classify as `American film actors (only)` *only* if:

1. **No TV Keywords:** There is zero mention of "Television", "Series", "Voice actor", or "Soap".
2. **Era:** They are "Silent film actors" or "Golden Age" stars (died before 1950s TV boom) *unless* they are explicitly linked to Radio/TV.
3. **Prestige:** They are defined by "Academy Awards" and "Blockbusters" and their neighbors are other movie stars (e.g., Humphrey Bogart, Bette Davis).

> **Correction from Error Cases:**
> * **Case 1 (Godunov):** He was a ballet dancer/film actor. He did not have "Voice actor" or "Sitcom" tags. **Result: Film Only.**
> * **Case 7 (Bridgette Wilson):** She had "Miss Teen USA" (TV event) and "Soap opera actors" in text. **Result: Film + TV.**
> * **Case 6 (William Conrad):** Text said "American voice actors" and "Radio actors". **Result: Film + TV.** (The model incorrectly guessed Film Only).
> 
> 

---

### Summary Table of Key Differentiators

| Feature | If Present, Likely Label is... |
| --- | --- |
| **"People from Ontario/BC"** | **Canadian Actors** |
| **"Genie Award" / "Juno"** | **Canadian Actors** |
| **"OBE" / "Knighthood" / "London"** | **English Actors** |
| **"Tony Award" + "Soap Opera"** | **American TV & Stage** |
| **"Voice Actor"** | **American Film & TV** |
| **"Daytime Emmy"** | **American Film & TV** |
| **"Academy Award" (No TV/Voice keywords)** | **American Film (Only)** |
| **"Silent Film Actor"** | **American Film (Only)** |

### Final Sanity Check

Before finalizing, ask:

* *Does the text explicitly say "Voice actor"?*  If yes, do **not** choose "Film Only".
* *Does the text mention a Canadian Province?*  If yes, override all American signals.
* *Does the text mention "Ancestry" (e.g., Irish-American)?*  Ignore this for Nationality; look for "People from..." tags instead."""


refining_actor="""
### Classification Protocol

#### Step 1: The Strict Nationality Filter (The "Passport" Check)

**Priority: High.** If an actor fits here, **STOP**. Do not check American categories.

* **Rule A: Geography Overrides Neighbors.**
* Ignore who they collaborate with. Look *only* at the **Node Information** first.
* If the text says "People from [City], **Ontario / British Columbia / Quebec / Alberta / Manitoba**", classify as **Canadian**.
* If the text says "People from **London / UK / England / Scotland / Wales**", classify as **English**.


* **Rule B: The "Awards & Titles" Check.**
* **Canadian Signals:** "Genie Award" (Canadian Oscar), "Juno Award", "First Nations" (distinct from Native American), "Order of Canada".
* **English Signals:** "OBE", "CBE", "Knighthood", "Dame", "Royal Shakespeare Company", "Alumni of RADA/Central School of Speech".


* **Rule C: Beware the Ancestry Trap.**
* "Irish-American", "French Canadian American", or "Italian-American" means they are **American**. Do not classify them as English or Canadian based on ancestry alone.



> **Correction form Error Cases:** In Case 1, the model missed "People from Brampton, Ontario". In Case 2, it missed "First Nations". These are immediate triggers for **Canadian**.
---

#### Step 2: The "Stage & TV" Niche (The Theater Professional)

If Step 1 confirms they are American, check this category *before* the general Film/TV split.

* **Target Label:** `American television actors and American stage actors`
* **The Profile:** These are actors known for Broadway, Musicals, and Soap Operas/TV series, but *lacking* a massive Blockbuster Film presence.
* **Keywords to Watch:**
* "Broadway", "Theatre", "Tony Award", "Drama Desk Award", "Musical theatre".
* **AND** TV indicators: "Soap opera", "Daytime Emmy", "Series regular".


* **Negative Constraint:** If they are listed as "Academy Award winner" (Oscars) for *Acting*, they likely belong in a Film category, not here.

> **Cases:** Case 3 (Elizabeth Ashley) had "Tony Award winners". This keyword is a primary driver for this category.

---

#### Step 3: The "Film Only" vs. "Film + Television" Split

You must distinguish between a "Movie Star" and a "Working Actor".

**A. The "Film + Television" Actor (The Generalist)**
Classify as `American film actors and American television actors` if *any* of the following are true:

1. **Voice Acting:** The text contains "**American voice actors**". (Voice work is predominantly TV cartoons/video games).
2. **TV Formats:** Text contains "Sitcom", "Soap opera", "Game show", "Talk show", "Reality", "SNL", "Saturday Night Live".
3. **TV Awards:** "Emmy Award" (Primetime or Daytime), "Golden Globe for **Television**".
4. **The Social Circle (Tie-Breaker):** If the node text is ambiguous, look at the **Collaborators**. If they hang out with "American game show hosts", "Stand-up comedians" (who usually have TV sitcomes), or "TV Anchors", they are in this category.

**B. The "Film Only" Actor (The Purist)**
Classify as `American film actors (only)` *only* if:

1. **No TV Keywords:** There is zero mention of "Television", "Series", "Voice actor", or "Soap".
2. **Era:** They are "Silent film actors" or "Golden Age" stars (died before 1950s TV boom) *unless* they are explicitly linked to Radio/TV.
3. **Prestige:** They are defined by "Academy Awards" and "Blockbusters" and their neighbors are other movie stars (e.g., Humphrey Bogart, Bette Davis).

> **Cases:**
> * **Case 4 (Godunov):** He was a ballet dancer/film actor. He did not have "Voice actor" or "Sitcom" tags. **Result: Film Only.**
> * **Case 5 (Bridgette Wilson):** She had "Miss Teen USA" (TV event) and "Soap opera actors" in text. **Result: Film + TV.**
> * **Case 6 (William Conrad):** Text said "American voice actors" and "Radio actors". **Result: Film + TV.** (The model incorrectly guessed Film Only).
> 
> 

## The Film vs. Film+TV Split (The Strict Filter) ##

**Default Assumption:** `American film actors (only)`

**Only switch to `American film actors and American television actors` if the text meets one of these Strict Conditions:**

**Condition A: The "TV Keyword" Match**
Does the *Actor's own text* contain any of these specific terms?

1. **"Television" / "TV"**
2. **"Series" / "Sitcom" / "Miniseries"**
3. **"Soap opera"**
4. **"Emmy Award"** (Primetime or Daytime)
5. **"Game show" / "Talk show" / "Host"**
6. **"Western film actors"** (Analysis shows these consistently map to Film+TV in this dataset)
7. **"Film serial actors"**

**Condition B: The "Voice Actor" Probability**

* If text contains "**Voice actor**", lean **Film + TV**.
* *Exception:* If the text also emphasizes "Dancer" or "Model" (e.g., Adrienne King), default to Film (Only).



**Condition C: The "Neighbor" Restriction**

* **Do NOT** switch to Film+TV just because neighbors are "Comedians" or "Emmy winners" (e.g., Jennifer Aniston neighbors often stay Film Only).
* **Exception:** If the actor is **Older** (born before 1940) and neighbors are "**Western film actors**" or "**Film serial actors**", you may switch to Film+TV.

### Cheat Sheet for Common Pitfalls

| If the Text says... | And NO explicit "TV" word... | The Label is... | Why? |
| --- | --- | --- | --- |
| **"Comedian"** | (e.g., Kenan Thompson) | **Film (Only)** | "Comedian"  "TV Actor" in this dataset. |
| **"Child actor"** | (e.g., Kenan Thompson) | **Film (Only)** | "Child actor"  "TV Actor". |
| **"Model"** | (e.g., Pamela Gidley) | **Film (Only)** | "Model"  "TV Actor". |
| **"Vaudeville"** | (e.g., Virginia Mayo) | **Film (Only)** | Historic stage  TV. |
| **"Western film actors"** | (e.g., Doug McClure) | **Film + TV** | Westerns are a bridge genre in this dataset. |
| **"Voice actor"** | (e.g., Adrienne King) | **Check Context** | usually TV, but can be Film Only if "Dancer" present. |
---

### Summary Table of Key Differentiators

| Feature | If Present, Likely Label is... |
| --- | --- |
| **"People from Ontario/BC"** | **Canadian Actors** |
| **"Genie Award" / "Juno"** | **Canadian Actors** |
| **"OBE" / "Knighthood" / "London"** | **English Actors** |
| **"Tony Award" + "Soap Opera"** | **American TV & Stage** |
| **"Voice Actor"** | **American Film & TV** |
| **"Daytime Emmy"** | **American Film & TV** |
| **"Academy Award" (No TV/Voice keywords)** | **American Film (Only)** |
| **"Silent Film Actor"** | **American Film (Only)** |

### Final Sanity Check

Before finalizing, ask:

* *Does the text explicitly say "Voice actor"?*  If yes, do **not** choose "Film Only".
* *Does the text mention a Canadian Province?*  If yes, override all American signals.
* *Does the text mention "Ancestry" (e.g., Irish-American)?*  Ignore this for Nationality; look for "People from..." tags instead."""

"""Based on the comprehensive analysis of the 180+ errors (specifically the 90+ new errors where the model aggressively guessed "Film + TV" for actors who were actually "Film Only"), we have found the dataset's logic is far stricter than real-world intuition.

**The "Hidden Category" Reality:** The dataset relies on whether the specific Wikipedia category string `"American television actors"` existed for that page in 2009.

* **Comedians** (e.g., Kenan Thompson) often lack this tag  **Film (Only)**.
* **Models** (e.g., Pamela Gidley) often lack this tag  **Film (Only)**.
* **Voice Actors** (e.g., Adrienne King) sometimes lack this tag  **Film (Only)**.
* **Modern Film Stars** (e.g., Catherine Keener) with TV cameos lack this tag  **Film (Only)**.

The "Film + Television" label is reserved for actors with **explicit** TV career indicators.

### The "Explicit TV" Protocol (Strict Version)

#### Step 1: The Nationality Filter (Priority 1)

If the actor fits here, **STOP**.

* **Canadian:** Text contains "Ontario", "British Columbia", "Toronto", "Montreal", "Genie Award", "Juno Award".
* **English:** Text contains "London", "UK", "England", "Knighthood", "OBE", "Royal Shakespeare".
* *Correction:* "Irish-American" or "Italian-American" are **American**.



#### Step 2: The "Stage" Niche (Priority 2)

* **Target:** `American television actors and American stage actors`
* **Rule:** Text must have **BOTH** Stage keywords ("Broadway", "Theatre", "Tony") **AND** TV keywords ("Soap", "Daytime Emmy").

#### Step 3: The Film vs. Film+TV Split (The Strict Filter)

**Default Assumption:** `American film actors (only)`

**Only switch to `American film actors and American television actors` if the text meets one of these Strict Conditions:**

**Condition A: The "TV Keyword" Match**
Does the *Actor's own text* contain any of these specific terms?

1. **"Television" / "TV"**
2. **"Series" / "Sitcom" / "Miniseries"**
3. **"Soap opera"**
4. **"Emmy Award"** (Primetime or Daytime)
5. **"Game show" / "Talk show" / "Host"**
6. **"Western film actors"** (Analysis shows these consistently map to Film+TV in this dataset)
7. **"Film serial actors"**

**Condition B: The "Voice Actor" Probability**

* If text contains "**Voice actor**", lean **Film + TV**.
* *Exception:* If the text also emphasizes "Dancer" or "Model" (e.g., Adrienne King), default to Film (Only).



**Condition C: The "Neighbor" Restriction**

* **Do NOT** switch to Film+TV just because neighbors are "Comedians" or "Emmy winners" (e.g., Jennifer Aniston neighbors often stay Film Only).
* **Exception:** If the actor is **Older** (born before 1940) and neighbors are "**Western film actors**" or "**Film serial actors**", you may switch to Film+TV.

### Cheat Sheet for Common Pitfalls

| If the Text says... | And NO explicit "TV" word... | The Label is... | Why? |
| --- | --- | --- | --- |
| **"Comedian"** | (e.g., Kenan Thompson) | **Film (Only)** | "Comedian"  "TV Actor" in this dataset. |
| **"Child actor"** | (e.g., Kenan Thompson) | **Film (Only)** | "Child actor"  "TV Actor". |
| **"Model"** | (e.g., Pamela Gidley) | **Film (Only)** | "Model"  "TV Actor". |
| **"Vaudeville"** | (e.g., Virginia Mayo) | **Film (Only)** | Historic stage  TV. |
| **"Western film actors"** | (e.g., Doug McClure) | **Film + TV** | Westerns are a bridge genre in this dataset. |
| **"Voice actor"** | (e.g., Adrienne King) | **Check Context** | usually TV, but can be Film Only if "Dancer" present. |"""
refining_actor_iter2="""To further revise the categorization of an actor based on the collaboration network and initial categorizations, follow this structured method:

1. **Examine the Collaboration Network**:
   - **Analyze Co-occurrence Patterns**: Review actors who appear on the same Wikipedia page as the target actor. These are professional collaborators or peers.
   - **Identify Influential Collaborators**: Determine which collaborators are highly prominent or frequently connected. Their categories can guide the target actor's classification.
   - **Count Collaborations by Category**: Tally how many collaborators belong to each category. A strong majority suggests the target actor's likely category.

2. **Compare Initial Categorizations**:
   - **Consistency Check**: Verify if the initial categorization aligns with the majority of collaborators' categories.
   - **Majority Rule**: If 70% or more collaborators belong to one category, this strongly indicates the target actor's category.
   - **Pattern Recognition**: Look for consistent patterns - actors in similar career stages or geographic regions often share categories.

3. **Review Reasoning for Categorizations**:
   - **Justifications**: Evaluate the reasoning provided for collaborators' categorizations. Strong justifications validate those categories.
   - **Common Themes**: Identify recurring themes in reasoning (e.g., "primarily film work", "dual medium career", "English origin").
   - **Geographic Consistency**: Ensure geographic indicators (American, English, Canadian) are consistently interpreted.

4. **Cross-Reference Career Patterns**:
   - **Medium Analysis**: Compare the target actor's career mediums (film, television, stage) with collaborators' mediums.
   - **Geographic Alignment**: Verify if collaborators from the same geographic origin share the same category.
   - **Career Trajectory**: Similar career paths often indicate the same category classification.

5. **Weigh Category-Specific Evidence**:
   - **"American film actors (only)"**: Requires exclusive film work and no television/stage presence. Collaborators should also be primarily film-only.
   - **"American film actors and American television actors"**: Most common dual-medium category. Look for collaborators with similar dual careers.
   - **"American television actors and American stage actors"**: Less common but distinct. Collaborators should show television and theater focus.
   - **"English actors"**: Geographic priority. Even with varied mediums, English nationality is the key determinant.
   - **"Canadian actors"**: Similar to English actors - nationality is the primary factor.

6. **Consider Network Homophily**:
   - **Social Network Principle**: Actors in the same category co-occur more frequently on Wikipedia pages due to similar career paths and collaborations.
   - **Category Clustering**: If the target actor's collaborators cluster strongly in one category, the target likely belongs there too.
   - **Professional Circles**: Actors work within professional circles, leading to higher collaboration rates within the same category.

7. **Iterative Adjustment**:
   - **Re-Evaluate Initial Judgment**: Based on collaboration network analysis, reconsider the initial categorization.
   - **Revise if Necessary**: Adjust the category if network evidence strongly contradicts the initial classification.
   - **Confidence Level**: Higher agreement among collaborators increases confidence in the category choice.

8. **Handle Edge Cases**:
   - **Mixed Signals**: If collaborators are split between categories, prioritize geographic categories (English/Canadian) over medium-specific ones.
   - **Limited Collaborations**: If few collaborations exist, rely more heavily on the actor's individual Wikipedia information.
   - **Category Imbalance**: Remember that "American film actors" is more common (~66% of dataset), but don't default to it without evidence.

9. **Final Decision**:
   - **Best Fit Category**: Choose the category that best represents the actor considering both individual profile and collaboration network.
   - **Network Validation**: Ensure the chosen category aligns with the majority of collaborators' categories.
   - **Document the Revision**: Note how the collaboration network influenced the decision.

10. **Final Review**:
    - **Self-Check**: Review the analysis process and final categorization carefully.
    - **Identify Issues**: Check for factual errors, inconsistencies, or missing information.
    - **Network Consistency**: Verify that the categorization fits logically within the collaboration network.
    - **Fix if Needed**: Adjust the category if any issues are found to ensure accurate classification.

By systematically analyzing the collaboration network and considering both individual profiles and network patterns, you can refine and improve the categorization. This approach leverages the social network structure where actors with similar careers and origins naturally collaborate more frequently, making the network a valuable validation tool for classification.
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

refining5_3="""### Cora Second-Iteration Revision Protocol (v6.0 - The Ultimate Unified Edition)

You are now in the **second iteration** of classification. 

#### Valid Categories & Advanced Sub-Domain Dictionary:
- **Rule Learning** — logic clauses, decision trees (ID3, C4.5), feature subset selection, ILP, Ensembles/Bagging of trees.
- **Neural Networks** — Neural architectures, backpropagation, SOM (Self-Organizing Maps), PLUS 1990s statistical models: Gaussian Processes, Support Vector Machines (SVM).
- **Case Based** — CBR, nearest neighbor, k-NN, instance-based, **conceptual clustering**.
- **Genetic Algorithms** — Evolutionary computation, Genetic Programming (GP), crossover, mutation, neuro-evolution.
- **Computational Learning Theory** — General mathematical bounds, PAC learning, sample complexity, boosting theory/proofs.
- **Reinforcement Learning** — Q-learning, MDP, delayed reward, temporal difference, agent learning.
- **Probabilistic Methods** — Bayesian networks, HMM, EM algorithm, statistical clustering (Gaussian Mixtures).

---

#### Phase 1: The Graph-Structure Absolute Dictate (NO EXCEPTIONS)
Graph structural context overrides ambiguous text.
1. Count total neighbors (References + Citations).
2. Tally the Rank-1 initial categorizations of these neighbors.
3. Compute percentage: `(Dominant Category Count / Total) * 100%`.
4. **THE INVIOLABLE DICTATE**: If the dominant neighbor category achieves $\ge$ 70%, **YOU MUST REVISE** the target paper to match this dominant category. 
   - *SYSTEM OVERRIDE AGAINST EVASION*: You are **FORBIDDEN** from playing smart. Do not attempt to correct "dataset errors". Do not state "the text suggests otherwise". If the math yields $\ge 70\%$, you apply the category. This is a hard-coded system constraint.

#### Phase 2: Resolving Sub-Domain Ambiguities & Historical Quirks (If Consensus < 70%)
If forced to analyze the text, apply these strict look-up rules to resolve edge cases:

- **Data Corruption (The Mismatched Text Rule)**: If the Title and Abstract strongly contradict each other, compare BOTH to the neighbor papers. Completely ignore the mismatched text and base your decision solely on the text that aligns with the neighbors.
- **The Clustering Split**:
  - EM Algorithm / Gaussian Mixtures $\rightarrow$ **Probabilistic Methods**
  - Self-Organizing Maps (SOM) $\rightarrow$ **Neural Networks**
  - Conceptual Clustering $\rightarrow$ **Case Based**
- **The Interdisciplinary & Historical Split**:
  - Mentions of *Gaussian Processes* or *SVM* $\rightarrow$ **Neural Networks**.
  - Genetic Programming (GP) / Evolving trees $\rightarrow$ **Genetic Algorithms**.
  - Inductive Logic Programming (ILP) $\rightarrow$ **Rule Learning**.
  - *Neuro-evolution* (using GA to evolve Neural Networks) $\rightarrow$ **Genetic Algorithms**.
  - Using GA merely as an optimizer for *feature selection* in decision trees $\rightarrow$ **Rule Learning**.
- **The Theory vs. Specifics Split**:
  - Proving mathematical bounds or sample complexity *in general* or for Boosting $\rightarrow$ **Computational Learning Theory**.
  - Proving bounds *specifically bound to a neural architecture* (e.g., recurrent perceptron) $\rightarrow$ **Neural Networks**.

#### Phase 3: The Baseline Discard Rule (Anti-Pattern)
Authors often mention famous algorithms (e.g., Backpropagation, C4.5) merely as baselines. **Always classify based on the PROPOSED method. Completely ignore baseline or comparison algorithms.**

#### Phase 4: Anti-Degradation Output Protocol
Because multi-step reasoning can cause context forgetting, you MUST end your response exactly with these two lines. Do not add any punctuation, alternative categories, or explanations after the category name.

**Self-Check: Did I select exactly two valid categories from the 7 options? Yes.**
**Final Category: [Insert EXACTLY TWO Category Name with relevance scores from 0 to 1 separated by a comma Here]**"""

refining5_4="""
### Cora Second-Iteration Revision Protocol (v8.0 - The Ultimate Unified Edition)

**[🚨 CRITICAL SYSTEM DIRECTIVES - READ FIRST]**
1. **Target Rank-1 ONLY**: You MUST treat ONLY the Rank-1 category (the first one listed) in the input as the "Initial Categorization". Ignore any secondary categories completely.
2. **The Graph Absolute Dictate (NO EXCEPTIONS)**: Calculate explicit math first: `(Dominant Neighbor Category Count / Total Neighbors) * 100%`. 
   - If the dominant neighbor category achieves $\ge$ 70%, you **MUST REVISE** the target paper to match this dominant category. 
   - *SYSTEM OVERRIDE*: You are **FORBIDDEN** from playing smart. Do not attempt to correct "dataset errors" or claim "the text suggests otherwise." If the math says $\ge 70\%$, you switch the category. Period.
3. **The Baseline & Benchmark Shield**: 
   - *Baselines*: Always classify based on the **PROPOSED** method. Completely ignore baseline or comparison algorithms.
   - *Benchmarks*: Papers often test their algorithms on classic problems (e.g., Pole Balancing, Maze, Games). **NEVER** classify a paper as Reinforcement Learning just because of its test environment.

---

#### Valid Categories & Expanded Master Dictionary:
- **Rule Learning** — logic clauses, decision trees (ID3, C4.5), feature selection/reduction, ILP, Ensembles of trees (Bagging/Boosting).
- **Neural Networks** — Neural architectures, backpropagation, SOM (Self-Organizing Maps), PLUS 1990s statistical physics models: **Gaussian Processes**, **Support Vector Machines (SVM)**.
- **Case Based** — CBR, nearest neighbor, k-NN, instance-based, **conceptual clustering**.
- **Genetic Algorithms** — Evolutionary optimization, Genetic Programming (GP), crossover, mutation, fitness, **neuro-evolution**.
- **Computational Learning Theory** — Formal theoretical analysis, PAC learning, VC dimension, general bounds, sample complexity, Sauer's lemma, finite automata (DFA) learning, boosting bounds proofs.
- **Reinforcement Learning** — Agent learning via reward signals, Q-learning, MDP, policy, temporal difference (Excluding testbed environments).
- **Probabilistic Methods** — Bayesian reasoning, Bayesian networks, HMM, EM algorithm, **statistical clustering (Gaussian Mixtures)**.

---

#### Phase 1: Data Corruption Handling (The "Mismatched Text" Rule)
The Cora dataset contains known crawling errors where a paper's Title and Abstract belong to completely different topics. 
- If the Title and Abstract strongly contradict each other, compare BOTH to the themes of the neighbor papers. **Completely ignore the mismatched text** and base your decision solely on the aligned text.

#### Phase 2: The Strict Interdisciplinary Override Stack (If Consensus < 70%)
If forced to analyze the text, evaluate it against this strict hierarchy. The first one that triggers wins:

- **Priority 1 (The Theory Supremacy)**: If the paper provides *mathematical proofs of learnability, sample complexity, or error bounds* for ANY algorithm (even Decision Trees, MDPs, or Neural Nets), it belongs to **Computational Learning Theory**.
- **Priority 2 (The Clustering & Programming Split)**: 
  - EM Algorithm / Gaussian Mixtures $\rightarrow$ **Probabilistic Methods**
  - SOM (Self-Organizing Maps) $\rightarrow$ **Neural Networks**
  - Conceptual Clustering $\rightarrow$ **Case Based**
  - Genetic Programming (GP) / Evolving trees $\rightarrow$ **Genetic Algorithms**
  - Inductive Logic Programming (ILP) $\rightarrow$ **Rule Learning**
- **Priority 3 (Neuro-evolution)**: If the paper explicitly proposes an evolutionary algorithm (GA) to evolve Neural Networks topologies/weights, it MUST be classified as **Genetic Algorithms**.
- **Priority 4 (The NIPS/90s Rule)**: Mentions of *Gaussian Processes* or *Support Vector Machines (SVM)* MUST be classified as **Neural Networks**.
- **Priority 5 (GA as an Optimizer)**: If a paper uses GA merely as a tool to perform *feature selection/reduction* or evolve *decision trees*, it belongs to **Rule Learning**.

---

**[🚨 ANTI-DEGRADATION OUTPUT PROTOCOL - READ LAST]**
Because long and complex multi-step reasoning can cause format forgetting, your final output MUST end exactly with these two lines. Do not add any punctuation, alternative categories, or conversational filler after the final category name.

**Self-Check: Did I select exactly two valid categories? Yes.**
**Final Category: [Insert EXACTLY TWO Category Name with relevance scores from 0 to 1 separated by a comma Here]**
"""
refining5_5="""
### Cora Single-Label Classification Protocol  (v6.0 - The Ultimate Unified Edition)

You are tasked with a classification task. **CRITICAL RULE:** You must select **EXACTLY TWO** categories with relevance scores from 0 to 1 for the target paper. 

**ANTI-MODERN BIAS DIRECTIVE**: Suppress your modern (2020s) machine learning intuition. Adhere strictly to the historical 1990s NIPS-era boundaries defined below.

#### Valid Categories & Rigid "Historical Faction" Dictionary:
- **Neural Networks (The Historical "Black Hole")**:
  - *Biological Override*: ANY paper predicting "Protein Structure", "DNA Sequence", "Finding Genes", or "Sequence Assembly" belongs here, EVEN IF it uses Decision Trees or HMMs.
  - *Theoretical Exemption*: Papers calculating "VC dimension" or "PAC bounds" SPECIFICALLY and EXCLUSIVELY for Neural Networks/Perceptrons belong HERE.
  - *Methodological Include*: PCA, ICA, Blind Source Separation, Gaussian Processes, SVM, RBF, Bayesian Interpolation, and Mixtures of Experts.
- **Computational Learning Theory**: Mathematical limits of learning. 
  - *Trump Cards*: Active Learning, Query by Committee, Boosting, AdaBoost, ECOC, PAC learning (general), Mistake bounds, finite automata (DFA) learning, PAC-Bayes.
- **Rule Learning**: Extracting human-readable logic. Includes Decision Trees (unless applied to DNA), Theory Refinement (KBANN), ILP, Constructive Induction, and Ensembles/Bagging of trees. *(QUIRK: Computer architecture "Instruction Level Parallelism" (ILP) MUST go here).*
- **Case Based**: Lazy Learning, k-NN, instance-based. *Rule: Feature Selection/Weighting specifically designed to solve the curse of dimensionality for k-NN heavily leans here.*
- **Probabilistic Methods**: Bayesian networks, HMM, EM algorithm (strictly OUTSIDE the context of continuous Neural architectures, PCA/ICA, or Computational Biology).
- **Reinforcement Learning**: MDPs, Q-Learning, delayed rewards, policy iteration.
- **Genetic Algorithms**: Evolutionary computation, mutation, fitness, simulated breeding.

---

#### Step 1: The Anti-Artifact "Confrontation" Mechanism
- **Rule**: You MUST explicitly confront the Title vs. the Abstract. 
- **Action**: Only if there is a massive conceptual mismatch (e.g., Title is pure Math Theorem, Abstract is empirical Neural Net experiment), or the text is a garbled bibliography string, this is an ARTIFACT. YOU MUST IGNORE THE ABSTRACT. Do NOT trigger this for generic "Technical Report" titles.

#### Step 2: Level-0 God Card (Object vs. Tool & Ultimate Goal Analysis)
Classify based on the **Primary Object of Study** or the **Final Domain Goal**, not the localized tool. Before looking at specific methods or graphs, scan for these ultimate goals:
- *Theory God Card*: Is the ultimate goal proving mathematical bounds, Active Learning, Boosting, or ECOC? (e.g., If ECOC uses Decision Trees as a testbed -> **Computational Learning Theory**).
- *Biology God Card*: Is the ultimate goal finding genes/protein structures? (e.g., If HMM is used for DNA -> **Neural Networks**).
- If a God Card is triggered, this is your FINAL answer.

#### Step 3: Level-1 Trump Card (Methodological Tracing)
If Step 2 does not apply, use the text to identify the core algorithm based on the Dictionary. 
- Trace meta-techniques to their **BASE algorithm**: 
  - Feature Selection for Decision Trees -> **Rule Learning**. 
  - Feature Selection for distance metrics (k-NN) -> **Case Based**.
  - Genetic Algorithm pruning a Decision Tree -> **Rule Learning**.
- **DO NOT let the citation graph override strong textual/methodological evidence.**

#### Step 4: The Graph-Structure Tie-Breaker (LAST RESORT ONLY)
The citation network is highly susceptible to historical misclassification poisoning. 
- **Rule**: ONLY IF the text is missing, garbled (Artifact from Step 1), or irreducibly ambiguous without any Dictionary keywords, may you use the citation graph.
- **Action**: Tally the Rank-1 initial categorizations of the neighbors. Adopt the dominant neighbor category if it reaches $\ge$ 70%.

#### Step 5: Anti-Evasion Single-Label Output Protocol
Because multi-step reasoning often causes models to output multiple categories or over-rely on graphs, you MUST conclude your response with the following exact confirmation lines:

**Self-Check 1: Did I confront the Artifacts and classify based on the Ultimate Object (God Card) rather than the localized Tool? Yes.**
**Self-Check 2: Did I use the Graph only as a last resort, prioritizing Textual Supremacy? Yes.**
**Self-Check 3: Am I outputting exactly TWO categories? Yes.**
**Final Category: [Insert EXACTLY TWO Valid Categories Name with relevance scores from 0 to 1 separated by a comma Here]**
"""
refining_actor="""**CRITICAL GRAPH CONTEXT:** This dataset represents a **Heterophilic Graph**. Pure film actors frequently collaborate with TV actors. 
* **DO NOT** directly copy a collaborator's category to the target node.
* **CORRECT TOPOLOGY USE:** Topology must only be used as a semantic trigger to search your internal memory, or as a final statistical tie-breaker. It is STRICTLY SUBORDINATE to your World Knowledge.

---

### 🛑 ANTI-HALLUCINATION & INDUSTRY LORE RULES 🛑
1. **THE 2014 TIME-WALL (CRITICAL):** This dataset was created around 2014. **DO NOT** use any TV shows or movies that premiered in 2015 or later to justify a label.
2. **The "Non-Scripted" Exemption:** "American singers", "Stand-up comedians", "Models", and "Reality TV participants" frequently appear on television in Talk Shows, Variety Specials, or as themselves. **THIS DOES NOT MAKE THEM TV ACTORS.** A Television Actor strictly requires portraying fictional characters in scripted network series.
3. **The Mega-Star Veto:** For global Cinema Legends (both Modern A-listers and Golden Age Hollywood Legends), winning an Emmy for a special, or doing occasional voiceover work, DOES NOT make them a "Television Actor". Ignore topology noise for them.
4. **The Pre-1980s Character Actor Exemption:** While modern actors need a "Series Regular" role, actors active in the 1950s-1970s who were "Prolific Character Actors" (dozens of guest appearances in classic network shows) DO count as TV actors.

---

### Classification Protocol

#### Step 0: ABSOLUTE SHORT-CIRCUIT (The Source Origin Principle)
Scan the text for:
* "Naturalized citizens...", "American immigrants...", "British expatriates..."
* ANY explicit Foreign birthplaces or institutions (e.g., "People from South Shields", "London", "Toronto") EVEN IF mixed with US tags (e.g., "New Jersey actors").
**RULE:** Their **Original Foreign Birthplace** takes absolute precedence. Output `English actors` or `Canadian actors` immediately and STOP. 

#### Step 1: Supreme Nationality Resolution
* Scan for hidden Commonwealth signifiers (e.g., British military, Canadian honors).
* If Foreign -> Output `English actors` or `Canadian actors` and STOP. 
* If American -> Proceed to Step 2.

#### Step 2: The Chronological Reality Check 
* If the text explicitly states their **Death Year is before 1950**, LOCK as Film/Stage only. Disable Step 4.

#### Step 3: Profession Extraction & Mandatory Parametric Wake-up
**Extract Signals:**
* **TV:** "Television", "Series", "Sitcom", "Soap opera", "Emmy". 
* **Stage:** "Stage actors", "Broadway", "Tony Award".
* **Film:** "Academy Award", "Movie", "Film", "Oscar". (Note: Voice actors doing theatrical animated films count as Film).

**MANDATORY PARAMETRIC OVERRIDES (World Knowledge > Text):**
* **Override A (The Cinema Giant & Non-Scripted Veto):** If text/topology has TV signals, but memory confirms they are a pure Global Cinema Legend OR a Singer/Comedian/Reality Star famous only for unscripted TV -> REMOVE TV signal. Map to `Film only`.
* **Override B (The Character Actor Rescue):** If text is barren, but memory confirms they were a pre-2014 Series Regular OR a Prolific TV Guest Star/Character Actor -> FORCE TV signal.
* **Override C (The Hidden Film Rescue):** If text yields Stage + TV, but they acted in notable theatrical films -> override Stage to Film. Map to `Film + TV`.

**INTERMEDIATE MAPPING:**
* Has ONLY Stage -> `American film actors (only)`.
* Has TV + Stage (and NO major film role) -> `American television actors and American stage actors`.
* Has ONLY TV OR (Film + TV) -> `American film actors and American television actors`.
* Has Film + Stage (No TV) OR (Film only) -> `American film actors (only)`.

#### Step 4: ADVANCED TOPOLOGICAL REASONING (The Graph-Memory Trigger)
If Step 3 mapping is `Film only`, you must actively use the topology graph before finalizing:
* **Rule 4.1: The Empty Node Block:** Is the actor's text COMPLETELY blank or extremely sparse (<2 valid tags)? If YES, block topology. Default to `Film only`.
* **Rule 4.2: The Lone Collaborator Anchor:** If the target has 1 or 2 collaborators, look at the collaborator's name. Did the target actor guest-star in THAT collaborator's famous fictional TV show? If YES -> Upgrade to `Film + TV`.
* **Rule 4.3: The Voice Actor Cluster:** If the target has "Voice actors" AND collaborators also have "Voice actors", they belong to TV Animation -> Upgrade to `Film + TV`.
* **Rule 4.4: The Statistical Upgrade:** If memory yields absolutely nothing, but **≥ 33% of collaborators** (min. 2) have explicit TV keywords -> Upgrade to `Film + TV`. (Subject to Override A Mega-Star Veto).

---

### Mandatory Reasoning Format (Checklist)
You MUST format your reasoning exactly as follows before outputting the final answer:

- **Step 0 & 1 (Nationality):** [Evaluate Source Origin Principle. Lock if Foreign.]
- **Step 2 (Era):** [Check death year.]
- **Step 3 (Parametric Signals):** [List explicit text signals. Apply Overrides A/B/C using memory (check for Singer/Comedian unscripted trap). State intermediate mapping.]
- **Step 4 (Advanced Topology):** [Evaluate ONLY if Step 3 lacks TV. Apply Empty Node Block. Apply Lone Collaborator Anchor. Evaluate 33% math threshold. State final decision.]
- **Conclusion:** [Final synthesized deduction based purely on the strict mapping above.]

The final answer is: $\boxed{EXACT MATCH WITH CONCLUSION}$"""
refining_actor_iter2="""**CRITICAL WARNING - DATASET NOISE & TEMPORAL LOCK:** * This dataset represents a historical snapshot. Real-world facts from recent years (post-2010 TV shows) DO NOT MATTER. 
* **NO CHITCHAT:** You are strictly forbidden from writing conversational paragraphs. You must ONLY output the 4-step checklist and the final boxed answer. Do not apologize or explain.

---

### 🛑 THE PERIPHERAL ENTERTAINER LOCK & EXCEPTION 🛑
Scan BOTH the `Node Name` (e.g., "Actor Name (comedian)") and the keywords. If they include "Singers", "Musicians", "DJs", "Radio", "Comedians", "Dancers", "Ballet dancers", "Danseurs", "Voice actors", "Adult models", "Porn stars", "Exercise instructors", or "Martial artists":
* **Default Lock:** You MUST lock them to `American film actors (only)`. 
* **The "Genuine Acting" Exception:** IGNORE "Emmy Award" or "Tony Award" tags for these people (they usually win for music/specials), **UNLESS** your World Knowledge explicitly guarantees they won it for a PRE-2010 SCRIPTED TV ACTING ROLE (e.g., Cyndi Lauper in "Mad About You"). Only then may you map them to `American film actors and American television actors`.
* Topology CANNOT override this lock.

---

### Classification Protocol (Follow Sequentially)

#### Step 0: The 5-Layer Nationality Funnel
Nationality determination requires synthesizing text and World Knowledge. Follow these prioritized rules top-to-bottom:
* **Rule 0.1 (Vacuum Knowledge Rescue):** If the text has ZERO geographic, institutional, or nationality keywords (e.g., only "1882 births"), consult your Parametric World Knowledge. If you know historically they are English or Canadian, output that and **STOP**.
* **Rule 0.2 (Explicit American Professional):** If the text explicitly contains phrases like "American voice actors", "American child actors", or "American soap opera actors", they are **American**. This OVERRIDES any foreign city tags (e.g., "Toronto"). Proceed to Step 1.
* **Rule 0.3 (Deep Commonwealth Parsing & The Naturalized Trap):** Scan for implicit UK/CA tags (e.g., "Artists Rifles", "South Wales", "South Shields", "Bristol"). If these exist, OR if the text contains "Naturalized citizens of the United States" alongside a foreign origin (e.g., "London"), you MUST prioritize the foreign origin -> Map to `English` or `Canadian` and **STOP**.
* **Rule 0.4 (The Parametric VETO):** *CRITICAL OVERRIDE.* If Rule 0.3 points to a foreign origin, BUT your World Knowledge knows they are a core Hollywood child star (e.g., Veronica Cartwright) or grew up entirely in the US system, **VETO the text**, assume American, and proceed to Step 1.
* **Rule 0.5 (American Default):** If text contains "American" word roots, US states, or no rules above apply -> "American background, proceed."

#### Step 1: Base Profession Extraction & World Knowledge Rescue
Extract explicit professional signals from the Target Actor's text:
* **TV Signals:** "Television", "Series", "Sitcom", "Soap opera", "Dragnet".
* **Stage Signals:** "Stage actors", "Broadway", "Tony Award".
* **Film Signals:** "Academy Award", "Movie", "Film", "Oscar", "Western film actors".

**Intermediate Strict Mapping:**
* **Check Lock:** Apply the Peripheral Entertainer Lock & Exception first.
* If explicit TV signals exist -> Map to `American film actors and American television actors`.
* If ONLY Stage signals exist -> Check World Knowledge. If they are a legendary film/TV actor (e.g., Mary Wickes), rescue to `American film actors and American television actors`. Otherwise, `American television actors and American stage actors`.
* **The Pre-2010 Scripted TV Rescue:** If 0 TV/Stage/Film signals exist in the text, you MAY consult World Knowledge. If you are absolutely certain they were a regular cast member in PRE-2010 SCRIPTED TELEVISION, rescue them to `American film actors and American television actors`.
* If ONLY Film signals exist, OR 0 signals exist and the Rescue fails -> Map to `American film actors (only)`.

#### Step 2: Advanced Heterophilic Topology
Evaluate topology ONLY IF Step 1 mapped to `American film actors (only)` AND they are NOT locked by the Peripheral Entertainer Lock:
* **The Mathematical Upgrade:** Count Total Collaborators (Y). Count exactly how many have explicit TV keywords (Emmy, Soap Opera, Television, Sitcom) in their text (X). 
* Calculate the percentage: X / Y.
* **If and ONLY IF ≥ 30% (X/Y >= 0.30)** of the collaborators have explicit TV keywords, upgrade to `American film actors and American television actors`.
* **Absolute Enforcement:** If X/Y < 0.30, or if X=0, you MUST retain `American film actors (only)`. Do NOT use intuition to override this math. Do NOT write "considering their broad career".

---

### MANDATORY OUTPUT TEMPLATE
You must copy this EXACT structure. Do not add any extra sentences.

- **Step 0 (Nationality):** [State which Rule 0.1-0.5 applied. Decision]
- **Step 1 (Base Signals):** [List explicit tags. Is Peripheral Lock applied? Was TV Rescue used? State intermediate mapping]
- **Step 2 (Topological Math):** [Evaluate ONLY if Step 1 is Film only. Total Collabs = Y. TV Collabs = X. Ratio = Z%. Decision]
- **Conclusion:** [Exact category match]

The final answer is: $\boxed{EXACT MATCH WITH CONCLUSION}$
"""
refining_actor_iter2="""# Core Graph Network Laws
1. **Embrace Heterophily (Profession):** Film actors naturally link to TV/Stage actors. NEVER infer an acting medium purely because a neighbor possesses it (Reject the "Hub-Neighbor Fallacy" and "Topology Contagion").
2. **Embrace Homophily (Geography):** English and Canadian actors cluster tightly in subgraph networks. Topology can implicitly reveal nationality.
3. **Tolerate Meta-Noise:** The dataset contains missing tags and irrational geographic mapping errors. Do not overfit; follow the strict statistical heuristics below.
4. **Base-Class Gravity:** `American film actors (only)` is the absolute default baseline for this network. When in doubt or lacking explicit signals, fallback to this base.

# Adjudication Protocol (Strict Execution Order)

## Step 1: Geographic Deep Scan & Overrides
- **Rule 1A (Expatriate Absolute Override):** If keywords contain "expatriate actors in Canada" or "expatriates in the United Kingdom", lock to `Canadian actors` or `English actors` respectively, OVERRIDING any "American" tags.
- **Rule 1B (Compound American Override):** If Rule 1A does not apply, and keywords contain explicit American hybrid tags (e.g., "African American actors", "American voice actors", "American bloggers", "American comedians"), **LOCK** to American categories immediately. Ignore all raw "British", "Canadian", or "Toronto" tags.
- **Rule 1C (Explicit Geography):** If no American override exists, and keywords contain "English", "British", "Canadian", "London", or "Ontario", lock to `English actors` or `Canadian actors`.
- **Rule 1D (Implicit Topology):** If nationality is completely blank AND the actor was born **after 1960 (or is Living)**, use homophily: if their 1st-order collaborators are predominantly UK/Canadian, infer English/Canadian.

## Step 2: Quarantine Zones & Superstar Blackholes (Crucial Defenses)
Apply these strict overrides BEFORE looking at topological degrees for upgrades:
- **Rule 2A (The Superstar Blackhole):** If the node's Degree **> 15**, OR if they have "Academy Award" coupled with music tags.
  👉 **ACTION:** Their vast network creates extreme topological noise. Shield them from topology. Lock to `American film actors (only)`.
- **Rule 2B (The Unscripted / Periphery Quarantine):** If keywords contain ANY of the following: "television personalities", "hosts", "participants" (e.g., Idol), "reality television", "models", "comedians", "stunt performers", "Porn stars", "singers", "Musicians", "radio actors", "ballet dancers", "dancers", "politicians", "Vaudeville", or purely demographic terms ("Living people", "1980 births") WITH NO explicit TV acting tags:
  👉 **ACTION:** Quarantine the node. **Topology is permanently blocked.** Force fallback to `American film actors (only)`.
- **Rule 2C (The Voice Actor Threshold):** If the node contains `"voice actors"` or `"game show hosts"`:
  👉 **ACTION:** Check Degree. If Degree $\ge$ 3, override Stage and assign `American film actors and American television actors`. If Degree $\le$ 2, they are minor periphery dubbers; fallback to `American film actors (only)`.
- **Rule 2D (Strict Stage Precedence):** ONLY IF the node has `"Tony Award"` or `"Broadway"` AND lacks modern screen tags (Rule 2C) AND lacks a high degree/Academy Awards:
  👉 **ACTION:** Prioritize stage. Output `American television actors and American stage actors`.

## Step 3: Degree-Aware Spatio-Temporal Topology (For Sparse Nodes)
For American nodes that survived Step 2 (no quarantine, no ironclad signals), fuse Era with Degree to extract valid topological context:
- **Condition A (Golden Age Pipeline - ACTIVE TOPOLOGY):** Born between **1880 and 1929** (or marked as `"centenarians"` / `"Film serial actors"`) AND Degree $\ge$ 4.
  👉 **ACTION:** This provides topological proof of deep industry integration and TV crossover during the 1950s TV boom. Upgrade to `American film actors and American television actors`.
- **Condition B (Franchise Crossover):** Born before 1930 and keywords contain specific famous franchise titles (e.g., "Space Odyssey series", "Quatermass").
  👉 **ACTION:** Upgrade to `American film actors and American television actors`.
- **Condition C (Modern Strict Fallback - TOPOLOGY SHIELD):** Born after **1930** (or "Living people") with no explicit internal TV tags.
  👉 **ACTION:** Modern sparse nodes are usually minor film actors. Do NOT upgrade based on neighbors' TV tags. Fallback to `American film actors (only)`.

## Step 4: Adjudicate the 3 Channels
Evaluate the reasoning from the 3 input channels:
- **VETO** any channel that uses professional homophily (e.g., "Target is TV because collaborator X is TV").
- **VETO** any channel that promotes a Quarantined node (e.g., reality TV, models, singers, comedians) to TV/Stage actor.
- **ACCEPT** channels that strictly apply Geographical Homophily (Step 1) or Spatio-Temporal Degree thresholds (Step 3A).

## Step 5: Final Output
Provide a brief structural analysis (evaluating Niche, Degree, Era, and your critique of the channels), and conclude STRICTLY with the exact format below:

The final answer is: $\\boxed{[Your Selected Category]}$
"""
def generate_arxiv_prompts(include_options, arxiv_natural_lang_mapping, first_iter=True, use_instructions=False):
    
    keys=arxiv_natural_lang_mapping.keys()
    keys=[f'{key} ({arxiv_natural_lang_mapping[key]})' for key in ['cs.GT','cs.MA','cs.RO','cs.NE','cs.IR','cs.SI','cs.CY']]

    if include_options:
        
        arxiv_prompts = {
            'subcategory': (f"Further revise the initial categorization for the paper. Choose the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper from the following categories:{', '.join(keys)}.The predicted sub-category should be in the format 'cs.XX'.\n\n"+"{}\n\nNow, apply this method to the following paper." if use_instructions else ""+ " The predicted sub-category should be in the format 'cs.XX'. If multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n") if not first_iter else (f"Predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper based on the titles and abstracts of the paper itself as well as its references and citations. Choose from the following categories:{', '.join(keys)}. The predicted sub-category should be in the format 'cs.XX'."),
            'identifier': f"Please predict the most appropriate original arxiv identifier for the paper. Your answer should be chosen from {', '.join([key.lower() for key in arxiv_natural_lang_mapping.keys()])}. The predicted arxiv identifier should be in the format 'arxiv cs.xx'.",
            'natural language': "Please predict the 2 most appropriate category for the paper based on the titles and abstracts of the paper itself as well as its references and citations. Your answer should be chosen from {}.".format(', '.join(['"{}"'.format(arxiv_natural_lang_mapping[key]) for key in arxiv_natural_lang_mapping.keys()]))
        }
    else:
        #an alternate prompt you can try for the second iteration or later.      
        f"Further revise the initial categorization for the paper. Predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format 'cs.XX'.\n\n"+"{}\n\nNow, apply this method to the following paper. The predicted sub-category should be in the format 'cs.XX'. If multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n"
        arxiv_prompts = {
            'subcategory':  ("Further refine the initial categorization for the paper. Predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper. The predicted sub-category should be in the format 'cs.XX'.\n\n"+"{}\n\nNow, apply this method to the following paper.\n\n" if use_instructions else "") if not first_iter else "Please predict the 2 most appropriate arXiv Computer Science (CS) sub-category for the paper based on the titles and abstracts of the paper itself as well as its references and citations. The predicted sub-category should be in the format 'cs.XX'.",
            'identifier': "Please predict the most appropriate original arxiv identifier for the paper. The predicted arxiv identifier should be in the format 'arxiv cs.xx'.",
            'natural language': "Please predict the most appropriate category for the paper. Your answer should be chosen from {}.".format(', '.join(['"{}"'.format(arxiv_natural_lang_mapping[key]) for key in arxiv_natural_lang_mapping.keys()]))
        }
    return arxiv_prompts


def generate_system_prompt(source, arxiv_style="subcategory", include_options=False, exlain=False, comfirm=False, options=None, first_iter=True, use_instructions=False):
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
        'cora': ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Computational Learning Theory", "Reinforcement Learning", "Probabilistic Methods"]
    }
    if options:
        categories['cora'] = options
    arxiv_prompts = generate_arxiv_prompts(include_options, arxiv_natural_lang_mapping, first_iter=first_iter, use_instructions=use_instructions)
    
    prompts = {
        'arxiv': arxiv_prompts[arxiv_style]
    }

    if not first_iter:
        prompts['cora']= "Further revise the initial categorization for the paper. Choose the most appropriate category for the paper from the following categories:\n\n{}\n\n"+"{}\n\nNow, apply this method to the following paper." if use_instructions else "" + " If multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n"
    else:
        prompts['cora']="Please predict the 2 most appropriate categories for the paper based on the titles and abstracts of the paper itself as well as its references and citations. Choose from the following categories:\n\n{}"+"\n\n{}" if use_instructions else ""

    prompts['wisconsin']="Please predict the 2 most appropriate categories for the webpage based on the URL and category of the webpages which the target page has outbound links to or has inbound linked from (for the non-main-page or resource pages labeled as 'other' within these webpages, their content abstract will be attached in addition) as well as the link pattern (inbound and outbound hyperlinks) of the target webpage. Choose from the following categories:\n\n{}"+"\n\n{}" if use_instructions else ""
    
    # Actor dataset prompts
    if not first_iter:
        prompts['actor'] = """# Role and Objective
You are an advanced Graph Neural Network (GNN) Expert Agent and the Final Adjudicator in a multi-agent debate system. Your task is to classify actor nodes in the HeTGB-Actor dataset into one of 5 categories based on text attributes, graph topology (Collaboration Count/Degree and Era), and to critically evaluate 3 initial reasoning channels.

# Core Graph Network Laws
1. **Embrace Heterophily (Profession):** Film actors naturally link to TV/Stage actors. NEVER infer an acting medium purely because a neighbor possesses it (Reject the "Hub-Neighbor Fallacy" and "Topology Contagion").
2. **Embrace Homophily (Geography):** English and Canadian actors cluster tightly in subgraph networks. Topology can implicitly reveal nationality.
3. **Tolerate Meta-Noise:** The dataset contains missing tags and irrational geographic mapping errors. Do not overfit; follow the strict statistical heuristics below.
4. **Base-Class Gravity:** `American film actors (only)` is the absolute default baseline for this network. When in doubt or lacking explicit signals, fallback to this base.Further revise the initial categorization for the actor. Choose the most appropriate category for the actor based on their career pattern, geographic origin, and collaboration network. Choose from the following categories:\n\n{}\n\n"""+("{}\n\nNow, apply this method to the following actor." if use_instructions else "")+" If multiple options apply, ensure these options are sorted from the most relevant to the least relevant.\n\n"
    else:
        prompts['actor'] = """### System Prompt: Actor Category Prediction Network

You are a STRICT, DETERMINISTIC graph-reasoning engine augmented with DEEP PARAMETRIC WORLD KNOWLEDGE. Your job is to classify the target actor based EXCLUSIVELY on their Wikipedia Node text (INCLUDING the Node Name) and their Topological Context (Collaborators). 

**CRITICAL WARNING - DATASET NOISE & TEMPORAL LOCK:** * This dataset represents a historical snapshot. Real-world facts from recent years (post-2010 TV shows) DO NOT MATTER. 
* **NO CHITCHAT:** You are strictly forbidden from writing conversational paragraphs. You must ONLY output the 4-step checklist and the final boxed answer. Do not apologize or explain.

Please predict the most appropriate category for the actor from these options ONLY:\n\n{}"""+"\n\n{}" if use_instructions else "Please predict the most appropriate category for the actor from these options ONLY:\n\n{}"
       
    prompt = prompts[source]

    if source in ['cora']:
        categories_list = "\n".join(categories[source])
        if first_iter:
            if use_instructions:
                return prompt.format(categories_list,refining3_3)
            return prompt.format(categories_list)
        else:
            if use_instructions:
                return prompt.format(categories_list, refining5_5)
            return prompt.format(categories_list)
    elif source == 'wisconsin':
        if use_instructions:
            return prompt.format("\n".join(['faculty', 'staff', 'department', 'course', 'project', 'student', 'other']), refining_wisconsin)
        return prompt.format("\n".join(['faculty', 'staff', 'department', 'course', 'project', 'student', 'other']))
    elif source == 'actor':
        actor_categories = ["American film actors (only)", 
                           "American film actors and American television actors", 
                           "American television actors and American stage actors", 
                           "English actors", 
                           "Canadian actors"]
        if first_iter:
            if use_instructions:
                return prompt.format("\n".join(actor_categories), refining_actor)
            return prompt.format("\n".join(actor_categories))
        else:
            if use_instructions:
                return prompt.format("\n".join(actor_categories), refining_actor_iter2)
            return prompt.format("\n".join(actor_categories))
    elif source == 'arxiv':
        if first_iter:
            if use_instructions:
                return prompt+"\n\n"+refining3
            return prompt
        else:
            if use_instructions:
                return prompt.format(refining5)
            return prompt

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
    