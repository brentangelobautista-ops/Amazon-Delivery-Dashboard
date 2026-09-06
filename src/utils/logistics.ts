export const haversine = (lat1:number, lon1:number, lat2:number, lon2:number) => { const r=6371, rad=(n:number)=>n*Math.PI/180; const a=Math.sin(rad(lat2-lat1)/2)**2+Math.cos(rad(lat1))*Math.cos(rad(lat2))*Math.sin(rad(lon2-lon1)/2)**2; return 2*r*Math.asin(Math.sqrt(a)) }
export const median=(values:number[])=>{const s=[...values].sort((a,b)=>a-b), n=s.length; return n?s.length%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2:0}
export const avg=(values:number[])=>values.length?values.reduce((a,b)=>a+b,0)/values.length:0
export const formatDate=(date:Date)=>date.toLocaleDateString('en-US',{month:'short',day:'numeric'})
export const periodFor=(hour:number) => hour<12?'Morning':hour<17?'Afternoon':hour<21?'Evening':'Night'
export const quantile=(values:number[], q:number)=>{const sorted=[...values].sort((a,b)=>a-b);if(!sorted.length)return 0;const position=(sorted.length-1)*q,lower=Math.floor(position),upper=Math.ceil(position);return sorted[lower]+(position-lower)*(sorted[upper]-sorted[lower])}
export const pearson=(pairs:{x:number;y:number}[])=>{if(pairs.length<2)return 0;const mx=avg(pairs.map(p=>p.x)),my=avg(pairs.map(p=>p.y));const denominator=Math.sqrt(pairs.reduce((s,p)=>s+(p.x-mx)**2,0)*pairs.reduce((s,p)=>s+(p.y-my)**2,0));return denominator?pairs.reduce((s,p)=>s+(p.x-mx)*(p.y-my),0)/denominator:0}
