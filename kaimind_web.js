// kaimind_web.js — Kaimind in the browser: BPE tokenizer + GPT forward pass in plain JavaScript. No libraries.
// Loads window.KAIMIND (from kaimind_model.js, made by export_web.py).
(function(){
"use strict";
const KM={};
// ───────── tokenizer ─────────
const WORD_RE=/ ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;
function makeTok(merges){ const rank=new Map(merges.map((p,i)=>[p[0]+","+p[1],i])); const vocab=[]; for(let i=0;i<256;i++) vocab.push(new Uint8Array([i])); merges.forEach(([a,b])=>{ const v=new Uint8Array(vocab[a].length+vocab[b].length); v.set(vocab[a]); v.set(vocab[b],vocab[a].length); vocab.push(v); });
  const cache=new Map(); const enc=new TextEncoder(), dec=new TextDecoder();
  function encWord(w){ if(cache.has(w)) return cache.get(w); let ids=Array.from(enc.encode(w));
    while(ids.length>=2){ let best=null,br=null; for(let j=0;j<ids.length-1;j++){ const r=rank.get(ids[j]+","+ids[j+1]); if(r!==undefined&&(br===null||r<br)){ best=[ids[j],ids[j+1]]; br=r; } } if(best===null) break;
      const out=[]; for(let j=0;j<ids.length;){ if(j<ids.length-1&&ids[j]===best[0]&&ids[j+1]===best[1]){ out.push(256+br); j+=2; } else { out.push(ids[j]); j++; } } ids=out; }
    cache.set(w,ids); return ids; }
  return { encode(t){ const out=[]; for(const w of (t.match(WORD_RE)||[])) out.push(...encWord(w)); return out; },
           decode(ids){ let n=0; ids.forEach(i=>n+=vocab[i].length); const b=new Uint8Array(n); let o=0; ids.forEach(i=>{ b.set(vocab[i],o); o+=vocab[i].length; }); return dec.decode(b); } }; }
// ───────── weights ─────────
function unpack(t){ const raw=atob(t.b64); const n=raw.length; const bytes=new Uint8Array(n); for(let i=0;i<n;i++) bytes[i]=raw.charCodeAt(i);
  if(t.dtype==="f32") return new Float32Array(bytes.buffer);
  const q=new Int8Array(bytes.buffer); const f=new Float32Array(q.length); for(let i=0;i<q.length;i++) f[i]=q[i]*t.scale; return f; }
// ───────── model ─────────
function build(M){ const cfg=M.cfg, P={}; for(const k in M.tensors) P[k]=unpack(M.tensors[k]); const d=cfg.d_model, H=cfg.n_head, hd=d/H, V=cfg.vocab_size;
  const ln=(x,g,b,T)=>{ const y=new Float32Array(T*d); for(let t=0;t<T;t++){ let mu=0; for(let i=0;i<d;i++) mu+=x[t*d+i]; mu/=d; let v=0; for(let i=0;i<d;i++){ const z=x[t*d+i]-mu; v+=z*z; } const r=1/Math.sqrt(v/d+1e-5); for(let i=0;i<d;i++) y[t*d+i]=g[i]*((x[t*d+i]-mu)*r)+b[i]; } return y; };
  const mm=(x,W,b,T,inD,outD)=>{ const y=new Float32Array(T*outD); for(let t=0;t<T;t++){ const yo=t*outD; if(b) for(let j=0;j<outD;j++) y[yo+j]=b[j]; for(let i=0;i<inD;i++){ const xv=x[t*inD+i]; if(xv===0) continue; const wo=i*outD; for(let j=0;j<outD;j++) y[yo+j]+=xv*W[wo+j]; } } return y; };
  const gelu=v=>{ const c=Math.sqrt(2/Math.PI); return 0.5*v*(1+Math.tanh(c*(v+0.044715*v*v*v))); };
  function forward(ids){ const T=ids.length; let x=new Float32Array(T*d); for(let t=0;t<T;t++) for(let i=0;i<d;i++) x[t*d+i]=P.tok_emb[ids[t]*d+i]+P.pos_emb[t*d+i];
    for(let L=0;L<cfg.n_layer;L++){ const p=k=>P["l"+L+"."+k];
      const h=ln(x,p("ln1_g"),p("ln1_b"),T); const qkv=mm(h,p("Wqkv"),p("bqkv"),T,d,3*d); const y=new Float32Array(T*d); const sc=1/Math.sqrt(hd); const att=new Float32Array(T);
      for(let hh=0;hh<H;hh++){ const qo=hh*hd, ko=d+hh*hd, vo=2*d+hh*hd;
        for(let t=0;t<T;t++){ let mx=-Infinity; for(let s=0;s<=t;s++){ let a=0; for(let i=0;i<hd;i++) a+=qkv[t*3*d+qo+i]*qkv[s*3*d+ko+i]; a*=sc; att[s]=a; if(a>mx) mx=a; }
          let sum=0; for(let s=0;s<=t;s++){ att[s]=Math.exp(att[s]-mx); sum+=att[s]; } for(let s=0;s<=t;s++){ const w=att[s]/sum; for(let i=0;i<hd;i++) y[t*d+qo+i]+=w*qkv[s*3*d+vo+i]; } } }
      const o=mm(y,p("Wo"),p("bo"),T,d,d); for(let i=0;i<x.length;i++) x[i]+=o[i];
      const h2=ln(x,p("ln2_g"),p("ln2_b"),T); const m1=mm(h2,p("W1"),p("b1"),T,d,4*d); for(let i=0;i<m1.length;i++) m1[i]=gelu(m1[i]); const m2=mm(m1,p("W2"),p("b2"),T,4*d,d); for(let i=0;i<x.length;i++) x[i]+=m2[i]; }
    const hf=ln(x,P.lnf_g,P.lnf_b,T); const last=(T-1)*d; const logits=new Float32Array(V); for(let v=0;v<V;v++){ let s=0; const eo=v*d; for(let i=0;i<d;i++) s+=hf[last+i]*P.tok_emb[eo+i]; logits[v]=s; } return logits; }
  // incremental forward with a KV cache: one token at a time, attention over everything cached so far
  function newCache(){ return {K:[],V:[],n:0}; }
  function step(id,cache){ const t=cache.n; const x=new Float32Array(d); for(let i=0;i<d;i++) x[i]=P.tok_emb[id*d+i]+P.pos_emb[t*d+i];
    for(let L=0;L<cfg.n_layer;L++){ const p=k=>P["l"+L+"."+k]; if(!cache.K[L]){ cache.K[L]=new Float32Array(cfg.ctx*d); cache.V[L]=new Float32Array(cfg.ctx*d); }
      const h=ln(x,p("ln1_g"),p("ln1_b"),1); const qkv=mm(h,p("Wqkv"),p("bqkv"),1,d,3*d); cache.K[L].set(qkv.subarray(d,2*d),t*d); cache.V[L].set(qkv.subarray(2*d,3*d),t*d);
      const y=new Float32Array(d); const sc=1/Math.sqrt(hd); const att=new Float32Array(t+1); const K=cache.K[L], Vv=cache.V[L];
      for(let hh=0;hh<H;hh++){ const qo=hh*hd; let mx=-Infinity; for(let s=0;s<=t;s++){ let a=0; for(let i=0;i<hd;i++) a+=qkv[qo+i]*K[s*d+qo+i]; a*=sc; att[s]=a; if(a>mx) mx=a; }
        let sum=0; for(let s=0;s<=t;s++){ att[s]=Math.exp(att[s]-mx); sum+=att[s]; } for(let s=0;s<=t;s++){ const w=att[s]/sum; for(let i=0;i<hd;i++) y[qo+i]+=w*Vv[s*d+qo+i]; } }
      const o=mm(y,p("Wo"),p("bo"),1,d,d); for(let i=0;i<d;i++) x[i]+=o[i];
      const h2=ln(x,p("ln2_g"),p("ln2_b"),1); const m1=mm(h2,p("W1"),p("b1"),1,d,4*d); for(let i=0;i<m1.length;i++) m1[i]=gelu(m1[i]); const m2=mm(m1,p("W2"),p("b2"),1,4*d,d); for(let i=0;i<d;i++) x[i]+=m2[i]; }
    cache.n=t+1; const hf=ln(x,P.lnf_g,P.lnf_b,1); const logits=new Float32Array(V); for(let v=0;v<V;v++){ let s=0; const eo=v*d; for(let i=0;i<d;i++) s+=hf[i]*P.tok_emb[eo+i]; logits[v]=s; } return logits; }
  function sample(logits,temp,topk){ const V=logits.length; const lg=new Float64Array(V); for(let i=0;i<V;i++) lg[i]=logits[i]/Math.max(temp,1e-6);
    if(topk){ const sorted=Array.from(lg).sort((a,b)=>b-a); const kth=sorted[Math.min(topk,V)-1]; for(let i=0;i<V;i++) if(lg[i]<kth) lg[i]=-Infinity; }
    let mx=-Infinity; for(let i=0;i<V;i++) if(lg[i]>mx) mx=lg[i]; let sum=0; const pr=new Float64Array(V); for(let i=0;i<V;i++){ pr[i]=Math.exp(lg[i]-mx); sum+=pr[i]; } let r=Math.random()*sum; for(let i=0;i<V;i++){ r-=pr[i]; if(r<=0) return i; } return V-1; }
  return {cfg,P,forward,step,newCache,sample}; }
KM.load=function(M){ M=M||window.KAIMIND; const tok=makeTok(M.merges); const net=build(M); return {tok,net,step:M.step,valLoss:M.val_loss,cfg:M.cfg,
  // generate(promptIds, opts) → async, calls opts.onToken(text) as it goes; stops on opts.stop strings or maxNew tokens
  async generate(ids,opts){ opts=opts||{}; const max=opts.maxNew||60, temp=opts.temperature||0.8, topk=opts.topK||40, stops=opts.stop||["\nUser:","\n\n"]; ids=ids.slice(); let text=""; const out=[];
    const ctxIds=ids.slice(-(M.cfg.ctx-max)); const cache=net.newCache(); let logits=null;
    for(let i=0;i<ctxIds.length;i++){ logits=net.step(ctxIds[i],cache); if(i%8===7) await new Promise(r=>setTimeout(r,0)); }        // feed the prompt through the cache
    for(let i=0;i<max;i++){ const t=net.sample(logits,temp,topk); out.push(t); text=tok.decode(out); if(opts.onToken) opts.onToken(text);
      if(stops.some(s=>text.includes(s))||cache.n>=M.cfg.ctx) break; logits=net.step(t,cache); if(i%2===1) await new Promise(r=>setTimeout(r,0)); }
    let clean=text; stops.forEach(s=>{ const j=clean.indexOf(s); if(j>=0) clean=clean.slice(0,j); }); return clean.trim(); } }; };
window.Kaimind=KM;
})();
