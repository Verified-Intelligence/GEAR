// Scalar restarted-PDHG LP arithmetic from opus_decay/pdhg2.cpp.
// The common COO table and its Ruiz scaling are prepared once for the lane
// block; independent CPU columns are tiled one at a time for cache locality.
// Search only. Python rechecks every returned bound and ray rigorously.
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>
using V=std::vector<double>;
static double norm(const V& x){double s=0;for(double v:x)s+=v*v;return std::sqrt(s);}
struct Table {
 int n,m,nz;const int64_t *rr,*cc;const double *aa,*hh,*ll,*uu;
 V a,Dr,Dc,mid,rad,S,hs,lw,uw;double eta,hn;
 Table(int n_,int m_,int nz_,const int64_t* r,const int64_t* c,const double* av,
       const double* h,const double* l,const double* u,const double* seed):
  n(n_),m(m_),nz(nz_),rr(r),cc(c),aa(av),hh(h),ll(l),uu(u),a(nz),Dr(m,1),Dc(n,1),mid(n),rad(n),S(n),hs(m),lw(n),uw(n){
  for(int j=0;j<n;j++){mid[j]=.5*(ll[j]+uu[j]);rad[j]=uu[j]>ll[j]?.5*(uu[j]-ll[j]):1;}
  for(int k=0;k<nz;k++)a[k]=aa[k]*rad[cc[k]];
  for(int it=0;it<11;it++){
   V rn(m,0),cn(n,0);
   for(int k=0;k<nz;k++){double v=std::abs(a[k]);if(it<10){rn[rr[k]]=std::max(rn[rr[k]],v);cn[cc[k]]=std::max(cn[cc[k]],v);}else{rn[rr[k]]+=v;cn[cc[k]]+=v;}}
   for(int i=0;i<m;i++){rn[i]=rn[i]>0?std::sqrt(rn[i]):1;Dr[i]/=rn[i];}
   for(int j=0;j<n;j++){cn[j]=cn[j]>0?std::sqrt(cn[j]):1;Dc[j]/=cn[j];}
   for(int k=0;k<nz;k++)a[k]/=rn[rr[k]]*cn[cc[k]];
  }
  V Am(m,0);for(int k=0;k<nz;k++)Am[rr[k]]+=aa[k]*mid[cc[k]];
  for(int j=0;j<n;j++){S[j]=rad[j]*Dc[j];lw[j]=(ll[j]-mid[j])/S[j];uw[j]=(uu[j]-mid[j])/S[j];}
  for(int i=0;i<m;i++)hs[i]=(hh[i]-Am[i])*Dr[i];
  V v(n);for(int j=0;j<n;j++)v[j]=seed[j];
  for(int it=0;it<40;it++){double nv=std::max(norm(v),1e-300);for(double& x:v)x/=nv;v=tmv(mv(v));}
  eta=.95/std::max(std::sqrt(norm(v)),1e-12);hn=norm(hs);
 }
 V mv(const V& x)const{V o(m,0);for(int k=0;k<nz;k++)o[rr[k]]+=a[k]*x[cc[k]];return o;}
 V tmv(const V& y)const{V o(n,0);for(int k=0;k<nz;k++)o[cc[k]]+=a[k]*y[rr[k]];return o;}
};
static void lane(const Table& t,const double* obj,double c0,int K,int check,
 const double* z0,const double* y0,bool warm,bool halpern,double rho,double deadline,
 double* zout,double* yout,double* rout,double* yfinal,double* stats,double* trace,
 double stop=INFINITY,double accuracy=-1.){
 int n=t.n,m=t.m,nz=t.nz;auto rr=t.rr,cc=t.cc;auto aa=t.aa,hh=t.hh,ll=t.ll,uu=t.uu;
 const V &Dr=t.Dr,&mid=t.mid,&S=t.S,&hs=t.hs,&lw=t.lw,&uw=t.uw;
 auto mv=[&](const V& x){return t.mv(x);};auto tmv=[&](const V& y){return t.tmv(y);};
 V cs(n);double off=c0;for(int j=0;j<n;j++){cs[j]=obj[j]*S[j];off+=obj[j]*mid[j];}
 double eta=t.eta,hn=t.hn,omega=hn>1e-10?std::max(norm(cs),1e-10)/std::max(hn,1e-10):1;
 V w(n,0),y(m,0),ws(n,0),ys(m,0),wl(n,0),yl(m,0),besty(m,0),bestr(m,0);
 for(int j=0;j<n;j++)w[j]=std::clamp(0.,lw[j],uw[j]);
 if(warm){for(int j=0;j<n;j++)w[j]=std::clamp((z0[j]-mid[j])/S[j],lw[j],uw[j]);for(int i=0;i<m;i++)y[i]=std::max(0.,y0[i])/Dr[i];}
 wl=w;yl=y;if(halpern){ws=w;ys=y;}
 auto physical=[&](const V& yy){V yp(m);for(int i=0;i<m;i++)yp[i]=std::max(0.,Dr[i]*yy[i]);return yp;};
 auto value=[&](const V& yp,bool ray){V r(n,0);double s=ray?0:c0;if(!ray)for(int j=0;j<n;j++)r[j]=obj[j];for(int k=0;k<nz;k++)r[cc[k]]+=aa[k]*yp[rr[k]];for(int i=0;i<m;i++)s-=hh[i]*yp[i];for(int j=0;j<n;j++)s+=std::min(r[j]*ll[j],r[j]*uu[j]);return s;};
 auto kkt=[&](const V& ww,const V& yy){V p=mv(ww);for(int i=0;i<m;i++)p[i]=std::max(0.,p[i]-hs[i]);double primal=off;for(int j=0;j<n;j++)primal+=cs[j]*ww[j];return std::hypot(omega*norm(p),std::abs(primal-value(physical(yy),false)));};
 V zero(m,0);double baseline=value(zero,false);
 besty=physical(y);
 double elast=kkt(w,y),eprev=elast,best=value(besty,false),raybest=0;int since=0,restarts=0;
 auto report=[&](){V ya(m);for(int i=0;i<m;i++)ya[i]=ys[i]/std::max(1,since);for(const V* p:halpern?std::vector<const V*>{&ys}:std::vector<const V*>{&y,&ya}){V yp=physical(*p);double val=value(yp,false);if(val>best){best=val;besty=yp;}}V diff(m);for(int i=0;i<m;i++)diff[i]=std::max(0.,(halpern?ys[i]:y[i])-yl[i]);for(const V* p:{halpern?&ys:&y,&diff}){V yp=physical(*p);double mx=1;for(double x:yp)mx=std::max(mx,std::abs(x));for(double& x:yp)x/=mx;double val=value(yp,true);if(val>raybest){raybest=val;bestr=yp;}}};
 report();int it=0,ti=0;if(trace)trace[ti++]=best;
 for(;it<K && best<stop && !(accuracy>=0 && kkt(halpern?ws:w,halpern?ys:y)<=accuracy);){if(deadline>=0 && std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()>=deadline)break;V before_w=w,before_y=y;int count=std::min(check,K-it);for(int t=0;t<count;t++){
   V gt=tmv(y),wn(n),ex(n);double tp=eta/omega;for(int j=0;j<n;j++){wn[j]=std::clamp((w[j]/tp-(cs[j]+gt[j]))/(1/tp),lw[j],uw[j]);ex[j]=2*wn[j]-w[j];}
   V g=mv(ex);double a1=(since+1.)/(since+2.),a0=1./(since+2.);
   for(int i=0;i<m;i++){double yn=std::max(0.,y[i]+eta*omega*(g[i]-hs[i]));if(halpern){ys[i]=yn;y[i]=a1*(2*yn-y[i])+a0*yl[i];}else{y[i]=yn;ys[i]+=y[i];}}
   for(int j=0;j<n;j++){if(halpern){ws[j]=wn[j];w[j]=a1*(2*wn[j]-w[j])+a0*wl[j];}else{w[j]=wn[j];ws[j]+=w[j];}}since++;
  }it+=count;report();if(trace)trace[ti++]=best;
  V wa(n),ya(m);for(int j=0;j<n;j++)wa[j]=ws[j]/std::max(1,since);for(int i=0;i<m;i++)ya[i]=ys[i]/std::max(1,since);
  double ec=kkt(w,y),ea=kkt(wa,ya);V wc=halpern?ws:(ea<ec?wa:w),yc=halpern?ys:(ea<ec?ya:y);ec=halpern?kkt(wc,yc):std::min(ec,ea);
  bool art=since>=.36*it;
  bool restart=ec<=.2*elast||(ec<=.8*elast&&ec>eprev)||art;eprev=ec;
  V dw(n),dy(m);for(int j=0;j<n;j++)dw[j]=wc[j]-wl[j];for(int i=0;i<m;i++)dy[i]=yc[i]-yl[i];double nw=norm(dw),ny=norm(dy);
  if(restart&&nw>1e-10&&ny>1e-10)omega=std::exp(.5*std::log(std::max(ny,1e-300)/std::max(nw,1e-300))+.5*std::log(omega));
  if(restart){w=wc;y=yc;wl=w;yl=y;elast=kkt(w,y);eprev=elast;if(!halpern){std::fill(ws.begin(),ws.end(),0);std::fill(ys.begin(),ys.end(),0);}since=0;restarts++;}
  double error=kkt(halpern?ws:w,halpern?ys:y),gain=std::max(0.,best-baseline);
  if(rho>=0 && error<=rho*gain) break;
  if(accuracy>=0 && (error<=accuracy || (before_w==w && before_y==y)))break;
 }
 for(int j=0;j<n;j++)zout[j]=mid[j]+S[j]*(halpern?ws[j]:w[j]);for(int i=0;i<m;i++){yout[i]=besty[i];rout[i]=bestr[i];yfinal[i]=std::max(0.,Dr[i]*(halpern?ys[i]:y[i]));}
 stats[8]=it;stats[9]=(rho>=0 && kkt(halpern?ws:w,halpern?ys:y)<=rho*std::max(0.,best-baseline));
 stats[0]=best;stats[1]=restarts;stats[2]=kkt(halpern?ws:w,halpern?ys:y);stats[3]=omega;stats[4]=raybest;stats[5]=eta;stats[6]=norm(cs);stats[7]=norm(hs);
}

extern "C" void solve_lp(int B,int n,int m,int nz,const int64_t* rr,const int64_t* cc,
 const double* aa,const double* obj,const double* c0,const double* hh,const double* ll,const double* uu,
 const double* seed,int K,int check,int warm,int halpern,const double* z0,const double* y0,double rho,double deadline,
 double* z,double* best_y,double* ray,double* y,double* stats,double* trace){
 Table table(n,m,nz,rr,cc,aa,hh,ll,uu,seed);
 for(int b=0;b<B;b++)lane(table,obj+b*n,c0[b],K,check,z0+b*n,y0+b*m,warm,halpern,rho,deadline,
                         z+b*n,best_y+b*m,ray+b*m,y+b*m,stats+b*10,trace?trace+b*(1+(K+check-1)/check):nullptr);
}

// Independent node boxes, one single-thread CPU call. CUDA uses node_lanes.
extern "C" void solve_node_batch(int B,int n,int m,int nz,const int64_t* rr,const int64_t* cc,
 const double* aa,const double* obj,const double* c0,const double* hh,const double* ll,const double* uu,
 const double* seed,int K,int check,int warm,int halpern,const double* z0,const double* y0,double rho,double deadline,
 double* z,double* best_y,double* ray,double* y,double* stats,const double* stop,const double* accuracy){
 for(int b=0;b<B;b++){
  Table table(n,m,nz,rr,cc,aa+b*nz,hh+b*m,ll+b*n,uu+b*n,seed);
  lane(table,obj+b*n,c0[b],K,check,z0+b*n,y0+b*m,warm,halpern,rho,deadline,
       z+b*n,best_y+b*m,ray+b*m,y+b*m,stats+b*10,nullptr,stop[b],accuracy[b]);
 }
}

extern "C" double node_makespan(int B,int capacity,const double* stats){
 V loads(capacity,0.);
 for(int b=0;b<B;b++)*std::min_element(loads.begin(),loads.end())+=stats[b*10+8];
 return *std::max_element(loads.begin(),loads.end());
}
