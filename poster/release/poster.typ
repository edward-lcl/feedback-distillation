// V2 candidate. Coordinates retain V1's square, numbered narrative and hero plot.
#let navy = rgb("142D4C")
#let teal = rgb("008A83")
#let pale = rgb("EDF7F6")
#let mint = rgb("71DAD0")
#set document(title: "Before You Scale — COLM poster V2", author: ("Saksham Kapoor", "Henry Tran", "Edward Lue Chee Lip", "Charlotte Le"), description: "COLM 2026 Workshop on Efficient Reasoning poster. Edward affiliation: Xi'an Jiaotong-Liverpool University (XJTLU).")
#set page(width: 34in, height: 34in, margin: 0pt, fill: white)
#set text(font: "Arial", size: 28pt, fill: navy)
#set par(leading: 0.32em, spacing: 0pt)
#set block(spacing: 0pt)
#let panel(x,y,w,h,body, fill: white, stroke: none) = place(top+left, dx:x*1pt,dy:y*1pt,block(width:w*1pt,height:h*1pt,fill:fill,stroke:stroke,inset:28pt,radius:16pt,body))
#let label(t) = text(size:25pt,weight:"bold",fill:teal,t)
#let title(t) = text(size:40pt,weight:"bold",t)
#panel(0,0,2448,365,fill:navy)[
  #pad(left:28pt,top:10pt)[
    #text(size:25pt,fill:mint,weight:"bold")[COLM 2026 · WORKSHOP ON EFFICIENT REASONING]
    #v(15pt)
    #text(size:70pt,weight:"bold",fill:white)[Before You Scale:]
    #v(2pt)
    #text(size:54pt,weight:"bold",fill:white)[The Cost of Supervision Mismatch in PRM Distillation]
    #v(20pt)
    #text(size:29pt,fill:white)[Saksham Kapoor\*¹˒² · Henry Tran¹˒³ · Edward Lue Chee Lip¹˒⁴ · Charlotte Le¹]
    #v(12pt)
    #text(size:22pt,fill:rgb("D9E5EE"))[\*Presenter · ¹Algoverse · ²University of Maryland, College Park · ³UCLA]
    #linebreak()
    #text(size:22pt,fill:rgb("D9E5EE"))[⁴Xi’an Jiaotong-Liverpool University (XJTLU)]
    #v(23pt)
    #text(size:28pt,fill:mint)[A process reward model (PRM) is a verifier that scores reasoning steps.]
  ]
]
// Genuine source pixels, smaller placement: 192 effective ppi, no upsampling.
#place(top+left,dx:2110pt,dy:26pt,block(width:280pt,height:76pt,fill:white,inset:20pt,radius:14pt)[#image("assets/algoverse-extracted.png",width:216pt)])
#panel(50,390,2348,78)[#text(size:44pt,weight:"bold")[Why did a better labeling teacher fail to produce a better student verifier?]]
#panel(50,490,735,330,fill:pale)[
  #label[01  REFERENCE → TEACHER → LABELS]
  #v(16pt)
  #title[A better labeling teacher…]
  #v(18pt)
  MATH first-error F1
  #v(10pt)
  #text(size:55pt,weight:"bold")[0.726 → 0.777]
  #v(15pt)
  Teacher without → with a reference solution. Privilege helps in this MATH probe.
]
#panel(855,490,735,330,fill:rgb("F0F3F7"))[
  #label[02  LABELS → ANSWER-BLIND VERIFIER]
  #v(16pt)
  #title[…but no student advantage.]
  #v(18pt)
  Initial 1.5B student · ROC-AUC
  #v(10pt)
  #text(size:55pt,weight:"bold")[0.631 vs 0.641]
  #v(15pt)
  #text(size:26pt)[Privileged vs no-GT supervision. *No-GT = teacher without a reference solution.* Positive control detects coarse teacher quality.]
]
#panel(1660,490,738,330,fill:pale)[
  #label[03  CONTROLLED SUPERVISION CHECK]
  #v(16pt)
  #title[Hold the student fixed.]
  #v(18pt)
  #text(size:34pt,weight:"bold")[Qwen2.5-3B-Instruct]
  #v(15pt)
  #text(size:27pt)[400 GSM8K training traces; score-only binary loss; balanced sampling; 500 training steps. Evaluate 1,000 MATH solutions.]
]
#place(top+left,dx:799pt,dy:620pt,text(size:42pt,fill:teal)[→])
#place(top+left,dx:1604pt,dy:620pt,text(size:42pt,fill:teal)[→])
#panel(50,855,1575,850,stroke:2pt+teal)[
  #label[PROCESSBENCH · A BENCHMARK FOR LOCATING FIRST REASONING ERRORS]
  #v(14pt)
  #title[First-error targets improve ranking on ProcessBench-MATH]
  #v(18pt)
  #image("assets/figure_rerender.svg",width:1515pt)
]
#panel(1660,855,738,850,fill:navy)[
  #set text(fill:white)
  #text(size:25pt,fill:mint,weight:"bold")[WHAT CHANGES? · CONCEPTUAL EXAMPLE]
  #v(20pt)
  #text(size:43pt,weight:"bold")[Keep only the first flag.]
  #v(18pt)
  #text(size:28pt)[First-error targets (paper: “PB-format”). Raw → first-error: *problem, prefix and step text stay identical*.]
  #v(25pt)
  #set table(stroke:none,inset:12pt)
  #text(fill:navy,size:28pt)[
    #table(columns:(330pt,154pt,154pt),
      table.header(table.cell(fill:rgb("DDE7EC"))[*Same reasoning text*],table.cell(fill:rgb("DDE7EC"))[*Raw flag*],table.cell(fill:rgb("DDE7EC"))[*First only*]),
      table.cell(fill:pale)[Step 1: 2x = 6],table.cell(fill:pale)[0],table.cell(fill:pale)[0],
      table.cell(fill:rgb("F8E6DC"))[Step 2: x = 2],table.cell(fill:rgb("F8E6DC"))[*1*],table.cell(fill:rgb("F8E6DC"))[*1*],
      table.cell(fill:pale)[Step 3: x + 1 = 3],table.cell(fill:pale)[*1*],table.cell(fill:pale)[*0*],
    )
  ]
  #v(16pt)
  #text(size:26pt)[1 = error flag; 0 = not flagged. A later 0 *does not mean mathematically correct*.]
  #v(25pt)
  #text(size:29pt,fill:mint,weight:"bold")[Targets + sampling change together.]
  #v(12pt)
  #text(size:27pt)[Relabeling changes the positive/negative pools used for balanced sampling. Their separate effects are not isolated.]
  #v(22pt)
  #text(size:26pt,fill:rgb("D9E5EE"))[Feedback wording and score magnitude are unused by this score-only diagnostic. This is not a cosmetic text-format change.]
]
#panel(50,1735,900,450,fill:pale)[
  #label[HOW TO READ THE GOLD REFERENCE]
  #v(18pt)
  #text(size:45pt,weight:"bold")[88.8% of the measured gap]
  #v(14pt)
  #text(size:27pt)[No-GT improvement ÷ its raw-to-gold gap, using unrounded four-seed mean AUCs.]
  #v(14pt)
  #text(size:26pt)[(first-error − raw) / (gold − raw)]
  #v(20pt)
  #text(size:28pt)[Gold is a *same-source reference*, with input/segmentation differences: 2,082 vs 2,080 generated training steps.]
  #v(20pt)
  #text(size:27pt,weight:"bold")[Descriptive ratio—not the fraction of failure causally explained. No CI is inferred.]
]
#panel(985,1735,525,450,fill:rgb("F0F3F7"))[
  #text(size:24pt,fill:teal,weight:"bold")[BEFORE A RECIPE SWEEP]
  #v(18pt)
  #text(size:35pt,weight:"bold")[Check targets first.]
  #v(22pt)
  #text(size:28pt)[One 500-step run]
  #v(8pt)
  #text(size:42pt,weight:"bold")[≈0.3 GPU-hours]
  #v(22pt)
  #text(size:27pt)[Preliminary recipe sweep]
  #v(8pt)
  #text(size:35pt,weight:"bold")[≈15 GPU-hours]
  #v(22pt)
  #text(size:25pt)[Setup-specific paper estimates; not verified end-to-end savings or the cost of the full study.]
]
#panel(1540,1735,858,450)[
  #label[OBSERVED TEACHER PATTERN IN THESE PROBES]
  #v(14pt)
  #image("assets/figure_teacher.svg",width:798pt)
  #v(10pt)
  #text(size:25pt)[MATH (N=400): paired 95% bootstrap CI. GSM8K (N=50): no CI available here. OlympiadBench: separate protocol, not plotted.]
  #v(10pt)
  #text(size:25pt)[Dataset and protocol differ; no universal difficulty law is established.]
]
#panel(50,2215,1920,190)[
  #text(size:33pt,weight:"bold")[Improved first-error ranking; downstream reasoning gains not established.]
  #v(17pt)
  #text(size:25pt)[Off-policy step-error labels → a separate answer-blind verifier. The 1.5B-to-3B contrast is not a clean capacity ablation. No universal cause or state-of-the-art claim.]
  #v(15pt)
  #text(size:22pt)[We thank Algoverse AI Research. COLM 2026 · V2.]
]
#panel(2075,2195,323,230)[
  #text(size:24pt,weight:"bold")[Read the paper]
  #v(10pt)
  #link("https://openreview.net/forum?id=gKNwyhDxPk",image("assets/qr-paper.svg",height:155pt))
]
