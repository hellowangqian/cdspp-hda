function P = CDLPP(dataA,dataB,labelA,labelB,options)
optsA = options;
optsA.gnd = labelA;
Wa = constructW1(labelA);
optsB = options;
optsB.gnd = labelB;
Wb = constructW1(labelB);
dataA = double(dataA);
dataB = double(dataB);
numA = length(unique(labelA));
numB = length(unique(labelB));
% Wc = zeros(length(labelA),length(labelB));
% num_class = numA;
% for i = 1:num_class
%     Wc = Wc + double(labelA==i)'*double(labelB==i);
% end
Wc = zeros(length(labelA),length(labelB));
num_class = numA;
for i = 1:num_class
    Wc = Wc + double(labelA==i)'*double(labelB==i);
end

Dc1 = diag(sum(Wc,2));
Dc2 = diag(sum(Wc,1));

Wa = double(Wa);
Wb = double(Wb);
Wc = double(Wc);
Da = diag(sum(Wa,2));
Db = diag(sum(Wb,2));


La = Da - Wa + options.lambda*Dc1;
Lb = Db - Wb + options.lambda*Dc2;


Sla = dataA'*La*dataA;
Slb = dataB'*Lb*dataB;
Swc1 = dataA'*Wc*dataB;
Swc2 = dataB'*Wc'*dataA;

Sla = (Sla+Sla')/2;
Slb = (Slb+Slb')/2;
%Swc1 = (Swc1+Swc1')/2;
%Swc2 = (Swc2+Swc2')/2;

A = [Sla,zeros(size(Sla,1),size(Slb,2));zeros(size(Slb,1),size(Sla,2)),Slb];
B = [zeros(size(Swc1,1),size(Swc2,2)), Swc1; Swc2,zeros(size(Swc2,1),size(Swc1,2))];
B = (B+B')/2;
A = A + options.alpha*eye(size(A,2));
[P,Diag] = eigs(double(B),double(A),options.ReducedDim,'la',options);
for i = 1:size(P,2)
    if (P(1,i)<0)
        P(:,i) = P(:,i)*-1;
    end
end