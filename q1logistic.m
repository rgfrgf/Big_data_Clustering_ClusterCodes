x=lift_kg;
y=putt_m-mean(putt_m);%debias
%y=(putt_m-mean(putt_m))/(std(putt_m));
%for i=1:1:length(y)
    %if putt_m(i)<mean(putt_m)
    %    putt_mc(i)=0;
    %else 
     %   putt_mc(i)=1;
    %end
%end
%for i=1:1:length(y)
 %   if y(i)<mean(y)
  %      putt_mc(i)=0;
  %  else 
  %      putt_mc(i)=1;
  %  end
%end
Y=floor(putt_m);
x=lift_kg;
B = mnrfit(x,Y);
%xx=0:1:80;
zz=1./(1+exp(B(1)+B(2)*x));
plot(sort(x),sort(zz),'r')
hold on 
scatter(x,y,'b')
xlabel('lift(kg)')
ylabel('putt(m)')
legend('logistic regression','debiased data')
ssr=sum((y-zz).^2);
sst=sum((y).^2);
r2=1-ssr/sst;