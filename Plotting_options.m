%--- 1) Read data back from Excel ---%
T_samples = readtable('bates_prices.xlsx', 'Sheet', 'samples');
prices  = T_samples.OptionPrice;

T_sum    = readtable('bates_prices.xlsx', 'Sheet', 'summary');
meanP    = sum(prices)/length(prices);
trueP    = 0.2664;

%--- 2) Plot histogram with vertical lines ---%
figure;
histogram(prices, 50, ...
    'EdgeColor','black', 'FaceAlpha',0.7);
hold on;

% mean line (red dashed)
xline(meanP, '--', 'Color','r','LineWidth',2);

% true price line (blue solid)
xline(trueP, '-','Color','k','LineWidth',2);

hold off;
grid on;
xlabel('Option Price');
ylabel('Frequency');
set(gca,'FontSize',20);
legend({'Samples','Mean','True Price'}, 'Location','best');
