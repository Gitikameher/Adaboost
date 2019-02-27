g=zeros(20000,10);t=0;
g_test=zeros(10000,10);
ind=zeros(20000,10);
ind_test=zeros(10000,10);
for i=0:9
       ind(find(l==i),i+1)=1;
       ind(find(l~=i),i+1)=-1;
       ind_test(find(l_test==i),i+1)=1;
       ind_test(find(l_test~=i),i+1)=-1;
end
alpha_t=zeros(10,784,51);
T_idx=zeros(10,250,2);
eps=zeros(10,1);
weight=zeros(250,10);
train_err=zeros(250,10);
test_err=zeros(250,10);
margin=zeros(5,20000,10);
max_idx=zeros(250,10);
for t=1:250
    w_bar=sample_weights(ind,g);
    [blah max_idx(t,:)]=max(w_bar);
    alpha=ind.*w_bar;
    alpha_pres=zeros(20000,10);
    al_test=zeros(10000,10);
    for i=1:10
    [alpha_pres(:,i) idx1  idx2]=max_values(x,alpha(:,i));
    T_idx(i,t,1)= idx1;T_idx(i,t,2)= idx2;
    eps(i, 1)= transpose(w_bar(:,i))*(ind(:,i)~=alpha_pres(:,i))/sum(w_bar(:,i));
    weight(t,i)= 0.5*log((1-eps(i))/eps(i));    
    if idx2<51
        al_test(:,i)=(2*((x_test(:,idx1)-(idx2/50))>=0)-1);
    end
    if idx2> 50
        al_test(:,i)=(-2*((x_test(:,idx1)-(idx2/50))>=0)+1);
    end
    end
    g=g+weight(t,:).*alpha_pres;
    g_test=g_test+weight(t,:).*al_test;
    train_err(t,:)=calculate_errors(g, ind);
    test_err(t,:)=calculate_errors(g_test, ind_test);
    if (t==5)
        margin(1, :,:)=ind.*g;
    end
    if (t==10)
         margin(2, :,:)=ind.*g;  end
     if (t==50)
         margin(3, :,:)=ind.*g; end
     if (t==100)
         margin(4, :,:)=ind.*g; end
     if (t==250)
         margin(5, :,:)=ind.*g;
    end
end

function err=calculate_errors(g, l)
    g=2*(g>0)-1;
    err=1-(sum(g==l,1)/20000);    
end

function [al id1 id2]=max_values(x,alpha)
val=-inf;al=zeros(20000,1);
   for j=1:784
    for th=0:101
     if (th<51)
     if (sum((2*((x(:,j)-(th/50))>=0)-1).*alpha))>val
         val=sum((2*((x(:,j)-(th/50))>=0)-1).*alpha);
         id1=j;id2=th+1;al(:,1)=(2*((x(:,j)-(th/50))>=0)-1) ;
     end
     else
     if (sum(-(2*((x(:,j)-(th/50))>=0)+1).*alpha))>val
         val=sum(-(2*((x(:,j)-(th/50))>=0)+1).*alpha);
         id1=j;id2=th+1;al(:,1)=(-2*((x(:,j)-(th/50))>=0)+1);      
     end
    end
     end 
   end
end

function w_bar=sample_weights(y, g)
w_bar=exp(-y.*g);
end