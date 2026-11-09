import struct, numpy as np

class TC:
    def __init__(s, b, p=0): s.b=b; s.p=p
    def byte(s):
        v=s.b[s.p]; s.p+=1; return v
    def varint(s):
        r=0; sh=0
        while True:
            x=s.byte(); r|=(x&0x7f)<<sh; sh+=7
            if not x&0x80: return r
    def zz(s):
        v=s.varint(); return (v>>1)^-(v&1)
    def read_val(s,t):
        if t==1: return True
        if t==2: return False
        if t==3: return struct.unpack('b',bytes([s.byte()]))[0]
        if t in (4,5,6): return s.zz()
        if t==7:
            v=struct.unpack('<d',s.b[s.p:s.p+8])[0]; s.p+=8; return v
        if t==8:
            n=s.varint(); v=s.b[s.p:s.p+n]; s.p+=n; return v
        if t in (9,10):
            h=s.byte(); n=h>>4; et=h&0xf
            if n==15: n=s.varint()
            return [s.read_val(et) if et not in (1,2) else (s.byte()==1) for _ in range(n)]
        if t==11:
            n=s.varint()
            if n==0: return {}
            kv=s.byte(); kt,vt=kv>>4,kv&0xf
            return {s.read_val(kt):s.read_val(vt) for _ in range(n)}
        if t==12: return s.struct()
        raise ValueError(t)
    def struct(s):
        out={}; last=0
        while True:
            h=s.byte()
            if h==0: return out
            d=h>>4; t=h&0xf
            fid = last+d if d else s.zz()
            last=fid
            out[fid]=s.read_val(t)

def snappy(src):
    p=0; n=0; sh=0
    while True:
        x=src[p]; p+=1; n|=(x&0x7f)<<sh; sh+=7
        if not x&0x80: break
    out=bytearray()
    while p<len(src):
        tag=src[p]; p+=1; t=tag&3
        if t==0:
            ln=tag>>2
            if ln>=60:
                nb=ln-59; ln=int.from_bytes(src[p:p+nb],'little'); p+=nb
            ln+=1; out+=src[p:p+ln]; p+=ln
        else:
            if t==1:
                ln=((tag>>2)&7)+4; off=((tag>>5)<<8)|src[p]; p+=1
            elif t==2:
                ln=(tag>>2)+1; off=int.from_bytes(src[p:p+2],'little'); p+=2
            else:
                ln=(tag>>2)+1; off=int.from_bytes(src[p:p+4],'little'); p+=4
            st=len(out)-off
            if off>=ln: out+=out[st:st+ln]
            else:
                for i in range(ln): out.append(out[st+i])
    assert len(out)==n
    return bytes(out)

def rle_hybrid(buf, bw, count):
    out=[]; p=0
    while len(out)<count and p<len(buf):
        h=0; sh=0
        while True:
            x=buf[p]; p+=1; h|=(x&0x7f)<<sh; sh+=7
            if not x&0x80: break
        if h&1:
            ng=h>>1; nbytes=ng*bw
            bits=np.unpackbits(np.frombuffer(buf[p:p+nbytes],np.uint8),bitorder='little')
            p+=nbytes
            if bw==0: vals=np.zeros(ng*8,np.int64)
            else:
                vals=bits[:ng*8*bw].reshape(-1,bw).astype(np.int64) @ (1<<np.arange(bw,dtype=np.int64))
            out.extend(vals.tolist())
        else:
            rl=h>>1; nb=(bw+7)//8
            v=int.from_bytes(buf[p:p+nb],'little'); p+=nb
            out.extend([v]*rl)
    return np.array(out[:count],np.int64)

PT={1:'<i4',2:'<i8',4:'<f4',5:'<f8'}

def read(path, cols):
    b=open(path,'rb').read()
    n=struct.unpack('<I',b[-8:-4])[0]
    fm=TC(b,len(b)-8-n).struct()
    schema=fm[2]; rgs=fm[4]
    leaf={}
    for el in schema[1:]:
        leaf[el[4].decode()]=el
    res={c:[] for c in cols}
    for rg in rgs:
        for cc in rg[1]:
            md=cc[3]; name=b'.'.join(md[3]).decode()
            if name not in cols: continue
            ptype=md[1]; codec=md[4]; nvals=md[5]
            el=leaf[name]; optional = el.get(3,0)==1
            start=md.get(11, md[9]); 
            if 11 in md: start=min(md[9], md[11])
            t=TC(b,start); dictv=None; got=[]; total=0
            while total<nvals:
                ph=t.struct(); clen=ph[3]; data=b[t.p:t.p+clen]; t.p+=clen
                ptyp=ph[1]
                if ptyp==2:
                    raw = snappy(data) if codec==1 else data
                    dn=ph[7][1]
                    dictv=decode_plain(raw,ptype,dn)
                    continue
                if ptyp==0:
                    dh=ph[5]; nv=dh[1]; enc=dh[2]
                    raw = snappy(data) if codec==1 else data
                    q=0
                    if optional:
                        ln=struct.unpack('<I',raw[:4])[0]; dl=rle_hybrid(raw[4:4+ln],1,nv); q=4+ln
                    else: dl=np.ones(nv,np.int64)
                elif ptyp==3:
                    dh=ph[8]; nv=dh[1]; nnull=dh[2]; enc=dh[4]; dlen=dh[5]; rlen=dh[6]
                    comp=dh.get(7,True)
                    lv=data[:rlen+dlen]; body=data[rlen+dlen:]
                    body = snappy(body) if (codec==1 and comp) else body
                    dl = rle_hybrid(lv[rlen:rlen+dlen],1,nv) if optional and dlen else np.ones(nv,np.int64)
                    raw=body; q=0
                else:
                    raise ValueError(ptyp)
                nn=int(dl.sum())
                if enc==0: vals=decode_plain(raw[q:],ptype,nn)
                elif enc in (2,8):
                    bw=raw[q]; idx=rle_hybrid(raw[q+1:],bw,nn); vals=dictv[idx]
                else: raise ValueError(('enc',enc))
                full=np.full(nv,np.nan) if ptype in (4,5) else np.zeros(nv,vals.dtype if len(vals) else np.int64)
                if ptype not in (4,5) and nn<nv: full=full.astype(float); full[:]=np.nan
                full[dl==1]=vals
                got.append(full); total+=nv
            res[name].append(np.concatenate(got))
    return {c:np.concatenate(v) for c,v in res.items()}

def decode_plain(raw,ptype,n):
    if ptype in PT: return np.frombuffer(raw,PT[ptype],count=n)
    if ptype==0: return np.unpackbits(np.frombuffer(raw,np.uint8),bitorder='little')[:n].astype(bool)
    if ptype==6:
        out=[];p=0
        for _ in range(n):
            l=struct.unpack('<I',raw[p:p+4])[0]; out.append(raw[p+4:p+4+l]); p+=4+l
        return np.array(out,dtype=object)
    raise ValueError(ptype)
