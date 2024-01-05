function res = logarithmicSmoothing(x,win,n)
%logarithmic smoothing.
%   win的长度应该是奇数
Nw = length(win);
assert(mod(Nw,2)==1,"窗函数的长度应该是奇数");
if n>(Nw-1)/2
    pstart = n - (Nw-1)/2;
    wstart = 1;
else
    pstart = 1;
    wstart = (Nw-1)/2 - n + 2;
end
if (n+(Nw-1)/2)<=length(x)
    pend = n + (Nw-1)/2;
    wend = Nw;
else
    pend = length(x);
    wend = Nw - (n+(Nw-1)/2) + length(x);
end
res = x(pstart:pend,:)'*win(wstart:wend);
end