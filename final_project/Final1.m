function X = Final1(c)
    a = dir('DAILY');
    b = {a.name};
    for i = 1:c
        r = round(rand(1)*length(b));
        disp(b(r))
        X = read(b(r));
    end
end

%Want this to return the Matrix of High and Lows for file
function X = read(file)
    mat = xlsread(file);
    High = mat(2);
    Low = mat(3);
    len = length(High);
    X = zeros(len, 2);
    for i = 1:len
        X(i, 1) = High(i);
        X(i, 2) = Low(i);
    end
end

% sharpe_func and cov_func have to take the output of read_func and output
% scalars
function [sharpe_ratios, cor] = cor_sharpe(dirname, read_func, sharpe_func, cov_func)
    dirname = strcat(dirname, '*.csv');
    files = dir(dirname);
    len = length(files);
    cor = zeros(len, len);
    sharpe_ratios = zeros(len);
    for i = 1:len
        stock_i = read_func(files(i).name);
        sharpe_ratios(i) = sharpe_func(stock_i);
        for j = 1:len
            stock_j = read_func(files(j).name);
            cor(i,j) = cov_func(stock_i, stock_j);
        end
    end     
end

% y is a 1 x n array
function sharpe = gauss_gp(y)
    [k n] = size(y);
    x = 0:n;
    meanfunc = [];                    % empty: don't use a mean function
    covfunc = @covhelper;              % Squared Exponental covariance function
    likfunc = @likGauss;              % Gaussian likelihood
    hyp = struct('mean', [], 'cov', [0 0], 'lik', -1); % hyper params
    
    hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    
    xs = [n+1] % next time step
    
    [mu var] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
    
    sharpe = mu[1] / var[1];
    




