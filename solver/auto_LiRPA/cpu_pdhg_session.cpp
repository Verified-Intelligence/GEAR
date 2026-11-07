// Persistent CPU alignment backend; search only, all certificates checked in Python.
#include "cpu_pdhg.cpp"
struct Lane {
 V cs,w,y,ws,ys,wl,yl,besty,bestr,primal;double off,omega,elast,eprev,best,raybest=0,error,baseline;
 bool ignore_cut,halpern;int since=0,it=0,restarts=0;
 V physical(const Table& t,const V& v)const{V o(t.m);for(int i=0;i<t.m;i++)o[i]=std::max(0.,t.Dr[i]*v[i]);return o;}
 double value(const Table& t,const double* obj,double c0,const V& v,bool ray=false)const{
  V r(t.n,0);double s=ray?0:c0;if(!ray)for(int j=0;j<t.n;j++)r[j]=obj[j];
  for(int k=0;k<t.nz;k++)r[t.cc[k]]+=t.aa[k]*v[t.rr[k]];
  for(int i=0;i<t.m;i++)s-=t.hh[i]*v[i];for(int j=0;j<t.n;j++)s+=std::min(r[j]*t.ll[j],r[j]*t.uu[j]);return s;
 }
 double search_kkt(const Table& t,const double* obj,double c0,const V& ww,const V& yy)const{
  V p=t.mv(ww);for(int i=0;i<t.m;i++)p[i]=(ignore_cut&&i==t.m-1)?0:std::max(0.,p[i]-t.hs[i]);
  double primal=off;for(int j=0;j<t.n;j++)primal+=cs[j]*ww[j];
  return std::hypot(omega*norm(p),std::abs(primal-value(t,obj,c0,physical(t,yy))));
 }
 double kkt(const Table& t,const double* obj,double c0,const V& ww,const V& yy)const{
  // Assess the original enclosed table, not a rounded shifted surrogate.
  V z(t.n),p(t.m,0);double primal=c0;
  for(int j=0;j<t.n;j++){z[j]=std::clamp(t.mid[j]+t.S[j]*ww[j],t.ll[j],t.uu[j]);primal+=obj[j]*z[j];}
  for(int k=0;k<t.nz;k++)p[t.rr[k]]+=t.aa[k]*z[t.cc[k]];
  for(int i=0;i<t.m;i++)p[i]=(ignore_cut&&i==t.m-1)?0:std::max(0.,(p[i]-t.hh[i])*t.Dr[i]);
  return std::hypot(omega*norm(p),std::abs(primal-value(t,obj,c0,physical(t,yy))));
 }

 Lane(const Table& t,const double* obj,double c0,const double* z0,const double* y0,bool ignore,bool halp):
  cs(t.n),w(t.n),y(t.m),ws(t.n,0),ys(t.m,0),besty(t.m),bestr(t.m,0),off(c0),ignore_cut(ignore),halpern(halp){
  for(int j=0;j<t.n;j++){cs[j]=obj[j]*t.S[j];off+=obj[j]*t.mid[j];w[j]=std::clamp((z0[j]-t.mid[j])/t.S[j],t.lw[j],t.uw[j]);}
  for(int i=0;i<t.m;i++)y[i]=std::max(0.,y0[i])/t.Dr[i];wl=w;yl=y;
  if(ignore_cut && t.m)y.back()=yl.back()=0;
  if(halpern){ws=w;ys=y;}
  double hn=t.hn;if(ignore_cut){V h=t.hs;if(t.m)h.back()=0;hn=norm(h);}
  omega=hn>1e-10?std::max(norm(cs),1e-10)/std::max(hn,1e-10):1;
  primal=w;error=kkt(t,obj,c0,w,y);elast=eprev=search_kkt(t,obj,c0,w,y);besty=physical(t,y);best=value(t,obj,c0,besty);baseline=value(t,obj,c0,V(t.m,0));
  if(!halpern && baseline>best){best=baseline;std::fill(besty.begin(),besty.end(),0.);}
 }
 void advance(const Table& t,const double* obj,double c0,int count){
  for(int k=0;k<count;k++){
   V gt=t.tmv(y),wn(t.n),ex(t.n);double tp=t.eta/omega;
   for(int j=0;j<t.n;j++){wn[j]=std::clamp((w[j]/tp-(cs[j]+gt[j]))/(1/tp),t.lw[j],t.uw[j]);ex[j]=2*wn[j]-w[j];}
   V g=t.mv(ex);double a1=(since+1.)/(since+2.),a0=1./(since+2.);
   for(int i=0;i<t.m;i++){double yn=(ignore_cut&&i==t.m-1)?0:std::max(0.,y[i]+t.eta*omega*(g[i]-t.hs[i]));if(halpern){ys[i]=yn;y[i]=a1*(2*yn-y[i])+a0*yl[i];}else{y[i]=yn;ys[i]+=y[i];}}
   for(int j=0;j<t.n;j++){if(halpern){ws[j]=wn[j];w[j]=a1*(2*wn[j]-w[j])+a0*wl[j];}else{w[j]=wn[j];ws[j]+=w[j];}}since++;
  }it+=count;
  V ya(t.m),wa(t.n);for(int i=0;i<t.m;i++)ya[i]=ys[i]/std::max(1,since);for(int j=0;j<t.n;j++)wa[j]=ws[j]/std::max(1,since);
  for(const V* p:halpern?std::vector<const V*>{&ys}:std::vector<const V*>{&y,&ya}){V yp=physical(t,*p);double v=value(t,obj,c0,yp);if(v>best){best=v;besty=yp;}}
  V diff(t.m);for(int i=0;i<t.m;i++)diff[i]=std::max(0.,(halpern?ys[i]:y[i])-yl[i]);
  for(const V* p:{halpern?&ys:&y,&diff}){V yp=physical(t,*p);double mx=1;for(double v:yp)mx=std::max(mx,std::abs(v));for(double& v:yp)v/=mx;double v=value(t,obj,c0,yp,true);if(v>raybest){raybest=v;bestr=yp;}}
  double ec=search_kkt(t,obj,c0,w,y),ea=search_kkt(t,obj,c0,wa,ya);V wc=halpern?ws:(ea<ec?wa:w),yc=halpern?ys:(ea<ec?ya:y);ec=halpern?search_kkt(t,obj,c0,wc,yc):std::min(ec,ea);
  bool restart=ec<=.2*elast||(ec<=.8*elast&&ec>eprev)||since>=.36*it;eprev=ec;
  V dw(t.n),dy(t.m);for(int j=0;j<t.n;j++)dw[j]=wc[j]-wl[j];for(int i=0;i<t.m;i++)dy[i]=yc[i]-yl[i];double nw=norm(dw),ny=norm(dy);
  if(restart&&nw>1e-10&&ny>1e-10)omega=std::exp(.5*std::log(std::max(ny,1e-300)/std::max(nw,1e-300))+.5*std::log(omega));
  if(restart){w=wc;y=yc;wl=w;yl=y;elast=eprev=search_kkt(t,obj,c0,w,y);if(!halpern){std::fill(ws.begin(),ws.end(),0);std::fill(ys.begin(),ys.end(),0);}since=0;restarts++;}
  error=kkt(t,obj,c0,halpern?ws:w,halpern?ys:y);
  V cert(t.m);for(int i=0;i<t.m;i++)cert[i]=besty[i]/t.Dr[i];
  double pe=kkt(t,obj,c0,primal,cert),we=kkt(t,obj,c0,halpern?ws:w,cert),ae=kkt(t,obj,c0,wa,cert);
  if(we<pe){primal=halpern?ws:w;pe=we;}if(!halpern && ae<pe)primal=wa;
 }
};
struct Session {Table table;std::vector<Lane> lanes;const double *obj,*c0;
 Session(int B,int n,int m,int nz,int halpern,const int64_t* r,const int64_t* c,const double* a,const double* o,const double* off,
 const double* h,const double* l,const double* u,const double* seed,const double* z,const double* y,const uint8_t* mask):
 table(n,m,nz,r,c,a,h,l,u,seed),obj(o),c0(off){for(int b=0;b<B;b++)lanes.emplace_back(table,o+b*n,off[b],z+b*n,y+b*m,mask&&!mask[b],halpern);}
};
extern "C" void* session_new(int B,int n,int m,int nz,int halpern,const int64_t* r,const int64_t* c,const double* a,const double* o,const double* off,const double* h,const double* l,const double* u,const double* seed,const double* z,const double* y,const uint8_t* mask){return new Session(B,n,m,nz,halpern,r,c,a,o,off,h,l,u,seed,z,y,mask);}
extern "C" void session_delete(void* p){delete static_cast<Session*>(p);}
extern "C" void session_step(void* p,int count,int check,double* z,double* y,double* best_y,double* ray,double* candidate,double* stats){
 auto& s=*static_cast<Session*>(p);auto& t=s.table;
 for(size_t b=0;b<s.lanes.size();b++){auto& v=s.lanes[b];for(int k=0;k<count;k+=check)v.advance(t,s.obj+b*t.n,s.c0[b],std::min(check,count-k));
  for(int j=0;j<t.n;j++)z[b*t.n+j]=std::clamp(t.mid[j]+t.S[j]*(v.halpern?v.ws[j]:v.w[j]),t.ll[j],t.uu[j]);for(int i=0;i<t.m;i++){y[b*t.m+i]=std::max(0.,t.Dr[i]*(v.halpern?v.ys[i]:v.y[i]));best_y[b*t.m+i]=v.besty[i];ray[b*t.m+i]=v.bestr[i];}
  V mag(t.m,0);double gapmag=std::abs(s.c0[b]);
  for(int k=0;k<t.nz;k++)mag[t.rr[k]]+=std::abs(t.aa[k])*std::max(std::abs(t.ll[t.cc[k]]),std::abs(t.uu[t.cc[k]]));
  for(int i=0;i<t.m;i++)mag[i]=(mag[i]+std::abs(t.hh[i]))*t.Dr[i];
  for(int j=0;j<t.n;j++)gapmag+=std::abs(s.obj[b*t.n+j])*std::max(std::abs(t.ll[j]),std::abs(t.uu[j]));
  double ku=(2.*(t.nz+t.m+t.n)+16)*std::ldexp(1.,-53);
  double rounding=ku/(1-ku)*(v.omega*norm(mag)+gapmag);
  V cert(t.m);for(int i=0;i<t.m;i++)cert[i]=v.besty[i]/t.Dr[i];
  V wc=v.primal;int target=-1,count_nonzero=0;
  for(int j=0;j<t.n;j++)if(s.obj[b*t.n+j]!=0){target=j;count_nonzero++;}
  bool support=!v.ignore_cut && s.c0[b]==0 && count_nonzero==1 && std::abs(s.obj[b*t.n+target])==1;
  if(support){double point=std::clamp(s.obj[b*t.n+target]*v.value(t,s.obj+b*t.n,0,v.besty),t.ll[target],t.uu[target]);wc[target]=(point-t.mid[target])/t.S[target];}
  double ce=v.kkt(t,s.obj+b*t.n,s.c0[b],wc,cert),pe=v.kkt(t,s.obj+b*t.n,s.c0[b],v.primal,cert);
  if(pe<ce){wc=v.primal;ce=pe;}
  for(int j=0;j<t.n;j++)candidate[b*t.n+j]=std::clamp(t.mid[j]+t.S[j]*wc[j],t.ll[j],t.uu[j]);
  stats[b*6]=std::min(v.error,v.kkt(t,s.obj+b*t.n,s.c0[b],v.halpern?v.ws:v.w,cert));stats[b*6+1]=v.baseline;stats[b*6+2]=v.restarts;stats[b*6+3]=v.it;stats[b*6+4]=rounding;stats[b*6+5]=ce;
 }
}
extern "C" void session_compact(void* p,int B,const int64_t* ids,const double* obj,const double* c0){
 auto& s=*static_cast<Session*>(p);std::vector<Lane> kept;for(int b=0;b<B;b++)kept.push_back(std::move(s.lanes[ids[b]]));s.lanes=std::move(kept);s.obj=obj;s.c0=c0;
}
