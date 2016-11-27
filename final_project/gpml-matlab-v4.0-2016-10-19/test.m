function test

% Our code %
x = gpml_randn(0.8, 20, 1);                 % 20 training inputs
y = sin(3*x) + 0.1*gpml_randn(0.9, 20, 1);  % 20 noisy training targets
xs = linspace(-3, 3, 61)';                  % 61 test inputs

function [K,dK] = covhelper(Q, hyp, x, z)
[K,dK] = covSM(1,hyp,x,z);
end
meanfunc = [];                    % empty: don't use a mean function
covfunc = @covhelper;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);


hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
  fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
  hold on; plot(xs, mu); plot(x, y, '+')

%{
cor_sharpe('/home/nate/Courses/bigdata/bigdata_hw/final_project/stock_data', get_sharpe)


function [sharpe_ratios, cor] = cor_sharpe(dirname, sharpe_func)
    dirname = strcat(dirname, '*.csv');
    files = dir(dirname);
    len = length(files);
    cor = zeros(len, len);
    sharpe_ratios = zeros(len);
    for i = 1:len
        stock_i = read_file(files(i).name);
        sharpe_ratios(i) = sharpe_func(stock_i);
        for j = 1:len
            stock_j = read_file(files(j).name);
            cor(i,j) = corrcoeff(stock_i, stock_j);
        end
    end     
end


function sr = get_sharpe(stock)
    sr = 0;
end

function vec = read_file(file_name)
    vec = [1. 2. 3.]
end

end
%}
end