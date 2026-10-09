#let navy = rgb("142D4C")
#let teal = rgb("008A83")
#let pale = rgb("EDF7F6")
#let gray = rgb("526478")
#let mode = sys.inputs.at("layout", default: "a")
#set page(width: 34in, height: 34in, margin: 0pt, fill: white)
#set text(font: "Arial", size: 28pt, fill: navy)
#set par(leading: 0.35em, spacing: 9pt)
#set block(spacing: 0pt)
#let panel(x,y,w,h,body, fill: white, stroke: none) = place(top+left, dx:x*1pt,dy:y*1pt,block(width:w*1pt,height:h*1pt,fill:fill,stroke:stroke,inset:28pt,radius:16pt,body))
#let label(t) = text(size:25pt,weight:"bold",fill:teal,t)
#let title(t) = text(size:40pt,weight:"bold",t)
#let fig(path,w) = if path in ("",) { [] } else { image(path,width:w*1pt) }
#panel(0,0,2448,365,fill:navy)[
  #pad(left:28pt,top:10pt)[
    #text(size:25pt,fill:rgb("71DAD0"),weight:"bold")[COLM 2026 · WORKSHOP ON EFFICIENT REASONING]
    #v(15pt)
    #text(size:70pt,weight:"bold",fill:white)[Before You Scale:]
    #v(2pt)
    #text(size:54pt,weight:"bold",fill:white)[The Cost of Supervision Mismatch in PRM Distillation]
    #v(20pt)
    #text(size:29pt,fill:white)[Saksham Kapoor\*¹˒² · Henry Tran¹˒³ · Edward Lue Chee Lip¹˒⁴ · Charlotte Le¹]
    #v(9pt)
    #text(size:22pt,fill:rgb("D9E5EE"))[\*Presenter · ¹Algoverse · ²University of Maryland, College Park · ³UCLA]
    #linebreak()
    #text(size:22pt,fill:rgb("D9E5EE"))[⁴Xi’an Jiaotong-Liverpool University / CSU]
  ]
]
#panel(1960,26,430,96,fill:white)[#image("assets/algoverse-extracted.png",width:374pt)]
#if mode == "a" {
 panel(50,390,2348,78)[#text(size:44pt,weight:"bold")[Why did a better labeling teacher fail to produce a better student verifier?]]
 panel(50,490,735,330,fill:pale)[
   #label[01  REFERENCE → TEACHER → LABELS]
   #v(15pt)
   #title[Answer-aware labels help…]
   #v(15pt)
   MATH first-error F1
   #v(10pt)
   #text(size:55pt,weight:"bold")[0.726 → 0.777]
   #v(10pt)
   Reference solutions improve teacher error localization in a tractable difficulty window.
 ]
 panel(855,490,735,330,fill:rgb("F0F3F7"))[
   #label[02  LABELS → ANSWER-BLIND VERIFIER]
   #v(15pt)
   #title[…but not this verifier.]
   #v(15pt)
   Initial 1.5B student · ROC-AUC
   #v(10pt)
   #text(size:55pt,weight:"bold")[0.631 vs 0.641]
   #v(10pt)
   Privileged vs no-GT supervision. A positive control detects coarse teacher-quality differences.
 ]
 panel(1660,490,738,330,fill:pale)[
   #label[03  FIXED-3B SUPERVISION CHECK]
   #v(15pt)
   #title[Hold the student fixed.]
   #v(15pt)
   #text(size:34pt,weight:"bold")[Same GSM8K source · fixed 3B]
   #v(13pt)
   400 GSM8K training traces · 500-step LoRA/BCE · 1,000 MATH evaluation solutions. Compare raw, re-rendered, and gold labels.
 ]
 place(top+left, dx:799pt,dy:620pt,text(size:42pt,fill:teal)[→])
 place(top+left, dx:1604pt,dy:620pt,text(size:42pt,fill:teal)[→])
 panel(50,855,1575,755,stroke:2pt+teal)[
   #label[THE CENTRAL RESULT · PROCESSBENCH-MATH]
   #v(10pt)
   #title[Rendering convention recovers verification signal]
   #v(12pt)
   #image("assets/figure_rerender.svg",width:1515pt)
 ]
 panel(1660,855,738,755,fill:navy)[
   #text(fill:rgb("71DAD0"),size:25pt,weight:"bold")[READ THE COMPARISON CORRECTLY]
   #v(24pt)
   #text(fill:white,size:47pt,weight:"bold")[Same traces.\ Re-rendered targets.]
   #v(24pt)
   #text(fill:white,size:29pt)[No-GT rises from *0.6183* to *0.7366*; privileged rises from *0.5494* to *0.6762*.]
   #v(24pt)
   #text(fill:white,size:29pt)[These are four-seed mean ROC-AUCs. Small dots show individual seeds, not confidence intervals.]
   #v(24pt)
   #text(fill:rgb("71DAD0"),size:37pt,weight:"bold")[88.8% gap recovery\*]
   #v(12pt)
   #text(fill:white,size:25pt)[\*No-GT gap to gold closed: (PB − raw) / (gold − raw), using unrounded four-seed mean ROC-AUCs. Descriptive ratio; no CI.]
   #v(18pt)
   #text(fill:rgb("D9E5EE"),size:24pt)[Format mismatch explains much of this failure; residual label-content and source effects remain.]
 ]
 panel(50,1645,1460,530,fill:pale)[
   #label[WHAT CHANGES? · CONCEPTUAL LABEL EXAMPLE]
   #v(10pt)
   #title[Use the target’s first-error convention]
   #v(15pt)
   #image("assets/figure_convention.svg",width:1320pt)
 ]
 panel(1540,1645,858,510)[
   #label[SUPPORTING EVIDENCE · TEACHER SWEET SPOT]
   #v(10pt)
   #image("assets/figure_teacher.svg",width:798pt)
   #v(9pt)
   #text(size:25pt)[Privilege helps where the teacher needs a reference and can use it. This is teacher F1, not student ROC-AUC.]
 ]
 panel(50,2190,1070,210,fill:pale)[
   #text(fill:teal,size:24pt,weight:"bold")[BEFORE A BROADER RECIPE SWEEP]
   #v(12pt)
   #text(fill:navy,size:37pt,weight:"bold")[Run a format-matched diagnostic.]
   #v(12pt)
   #text(fill:navy,size:26pt)[Reported ≈0.3 GPU-h diagnostic vs ≈15 GPU-h sweep]
   #v(10pt)
   #grid(columns:(200pt,1fr),gutter:8pt,text(size:21pt)[Diagnostic ≈0.3],rect(width:14pt,height:12pt,fill:teal,stroke:none),text(size:21pt)[Sweep ≈15],rect(width:700pt,height:12pt,fill:navy,stroke:none))
 ]
 panel(1150,2190,850,210)[
   #text(size:24pt)[*Scope:* off-policy step-error labels → a separate answer-blind verifier. Not a clean 1.5B-to-3B capacity ablation; not a universal cause or a SOTA claim.]
   #v(12pt)
   #text(size:21pt)[We thank Algoverse AI Research. Paper: methods, uncertainty, and references.]
 ]
 panel(2040,2180,358,230)[
   #text(size:23pt,weight:"bold")[Read the paper]
   #v(10pt)
   #if "qr-paper.svg" in json("assets/available.json") { link("https://openreview.net/forum?id=gKNwyhDxPk", image("assets/qr-paper.svg",height:155pt)) }
 ]
} else {
 panel(50,405,2348,100)[#title[Better labels are not enough: check the supervision contract.]]
 panel(50,540,735,1350,fill:pale)[
   #label[QUESTION → EXPERIMENT]
   #v(25pt)
   #title[Teacher advantage]
   #v(25pt)
   Answer-aware MATH first-error F1: *0.726 → 0.777*.
   #v(30pt)
   #image("assets/figure_teacher.svg",width:670pt)
   #v(35pt)
   #title[Transfer null]
   #v(20pt)
   Initial 1.5B student: *0.631 vs 0.641 ROC-AUC* (privileged vs no-GT).
   #v(35pt)
   #title[Controlled diagnostic]
   #v(20pt)
   Fixed 3B student, same GSM8K source, same training path. Change supervision source and rendering.
 ]
 panel(835,540,1563,1350,stroke:2pt+teal)[
   #label[RESULT → MECHANISM]
   #v(25pt)
   #title[Re-rendering recovers the signal]
   #v(30pt)
   #image("assets/figure_rerender.svg",width:1495pt)
   #v(25pt)
   Four-seed mean ROC-AUC. Seed dots are not confidence intervals.
   #v(30pt)
   #image("assets/figure_convention.svg",width:1495pt)
 ]
 panel(50,1930,1510,450,fill:navy)[
   #text(fill:white,size:43pt,weight:"bold")[Spend on diagnosis before scaling.]
   #v(25pt)
   #image("assets/figure_cost.svg",width:1050pt)
 ]
 panel(1600,1930,798,450)[
   #label[SCOPE & TAKEAWAY]
   #v(20pt)
   Format mismatch explains much of this observed failure. Residual label-content and source effects remain.
   #v(20pt)
   *88.8%* = (PB − raw) / (gold − raw), using unrounded four-seed mean ROC-AUCs. Descriptive ratio; no CI.
   #v(20pt)
   No universal cause, capacity ablation, or SOTA claim.
 ]
}
