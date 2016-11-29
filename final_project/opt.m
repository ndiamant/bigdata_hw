%run convex optimization to find minimum of the objective
function X = opt(W, S, Ex)
    sharpe = objective(W, S, Ex);
end


%compute expected return and variance for given weights W
function ratio = objective(W, S, Ex)
    E = 0;
    var = 0;
    for i = 1:length(W)
       RVi = normpdf(x, Ex(i), S(i));
       E = E + W(i)*Ex(i);
       inner = 0; 
       for j = 1:length(W)
           if (i ~= j)
               RVj = normpdf(x, Ex(j), S(j));
               cor = corrcoef(RVi, RVj);
               inner = inner + w(i)*w(j)*S(i)*S(j)*cor;
           end
       end
       var = var + ((W(i)^2 * S(i)^2)*inner
    end
    ratio = E / var;
end