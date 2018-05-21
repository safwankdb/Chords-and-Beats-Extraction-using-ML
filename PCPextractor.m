file = 'c2.wav' ;
[y,Fs,bits] = wavread(file); 

N = length(y); 
t = (1/Fs)*(1:N); 
y_fft = abs(fft(y));            
y_fft = y_fft(1:N/2)  ;    
PCP = zeros(12, 1) ;
fref = 130.8 ;
M = zeros(N/2 , 1) ;
for l = 1:N/2,
    if l == 1,
      M(l) = -1 ;
    else,
      M(l) = mod(round(12*log2((Fs*((l-1)/N))/fref)), 12) ;
    endif
endfor 
for p = 1:12,
  PCP(p) = ((y_fft.^2)')*kroneckerDel(M, (p-1)*ones(N/2, 1)) ;
endfor

PCP = PCP./sum(PCP) 
