function loadBlocks(editor){

const blocks = editor.BlockManager;


// HERO BLOCK

blocks.add("ai-hero",{

label:"AI Hero",

category:"AI Website",

content:`

<section style="
padding:100px;
background:#0b1120;
color:white;
text-align:center;
">

<h1 style="
font-size:64px;
">
AI Vision Platform
</h1>


<p style="
font-size:22px;
opacity:.8;
">
Computer vision for the next generation of robotics.
</p>


<button style="
padding:15px 35px;
border-radius:8px;
background:#2563eb;
color:white;
border:none;
font-size:18px;
">
Explore Technology
</button>


</section>

`

});



// FEATURE CARDS

blocks.add("feature-grid",{

label:"Feature Grid",

category:"AI Website",

content:`

<section style="
display:grid;
grid-template-columns:repeat(3,1fr);
gap:20px;
padding:60px;
">


<div>
<h2>
Computer Vision
</h2>
<p>
Real-time object detection.
</p>
</div>


<div>
<h2>
Robotics
</h2>
<p>
Autonomous systems.
</p>
</div>


<div>
<h2>
Analytics
</h2>
<p>
Advanced insights.
</p>
</div>


</section>

`

});



// DASHBOARD

blocks.add("ai-dashboard",{

label:"AI Dashboard",

category:"AI Website",

content:`

<div style="
padding:40px;
background:#111827;
color:white;
border-radius:20px;
">


<h2>
Model Performance
</h2>


<div style="
display:flex;
gap:30px;
">


<div>
<h1>
98.7%
</h1>
Accuracy
</div>


<div>
<h1>
120 FPS
</h1>
Inference
</div>


<div>
<h1>
42
</h1>
Objects
</div>


</div>

</div>

`

});


}