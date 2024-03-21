use rugfield::gen_grf;
use peroxide::fuga::*;
use rayon::prelude::*;
use indicatif::{ProgressBar, ParallelProgressIterator};

fn main() {
    // Gaussian Random Field generation
    let x_min = 0f64;
    let x_max = 1f64;
    let n = 1000;

    let grf_vec = (0 .. n)
        .map(|_| gen_grf(x_min, x_max, 0.1, 100))
        .collect::<Vec<_>>();

    // Normalize
    let grf_max_vec = grf_vec
        .iter()
        .map(|grf| grf.max())
        .collect::<Vec<_>>();

    let grf_min_vec = grf_vec
        .iter()
        .map(|grf| grf.min())
        .collect::<Vec<_>>();

    let grf_max_mean = grf_max_vec.mean();
    let grf_max_std  = grf_max_vec.sd();
    let grf_max_sigma = grf_max_mean + 1.0 * grf_max_std;

    let grf_min_mean = grf_min_vec.mean();
    let grf_min_std  = grf_min_vec.sd();
    let grf_min_sigma = grf_min_mean - 1.0 * grf_min_std;

    let grf_scaled_vec = grf_vec.iter()
        .map(|grf| {
            grf.fmap(|x| (x - grf_min_sigma) / (grf_max_sigma - grf_min_sigma) * 2f64 - 1f64)
        }).collect::<Vec<_>>();

    let x = linspace_with_precision(x_min, x_max, 100, 3);
    let y_range = linspace_with_precision(x_min, x_max, 20, 1);

    // Integration
    let grf_int_vec = grf_scaled_vec
        .par_iter()
        .progress_with(ProgressBar::new(n as u64))
        .map(|grf| {
            let cs = cubic_hermite_spline(&x, grf, Quadratic);
            //y_range.fmap(|y| integrate(|x| cs.eval(x), (x_min, y), G7K15R(1e-4, 20)))
            y_range.fmap(|y| cs.integrate((x_min, y)))
        })
        .collect::<Vec<_>>();

    // Plot grf
    let samples = 4;
    let line_style = [LineStyle::Solid, LineStyle::Dotted, LineStyle::Dashed, LineStyle::DashDot];
    let line_style = line_style.iter().cycle().take(samples).cloned().collect::<Vec<_>>();
    let color = ["darkblue", "red", "darkgreen", "darkorange", "purple"];
    let color = color.iter().cycle().take(samples).cloned().collect::<Vec<_>>();
    let legends = (1 ..=samples).map(|i| format!(r"GRF$_{}$", i)).collect::<Vec<_>>();
    let legends_str = legends
        .iter()
        .map(|x| x.as_str())
        .collect::<Vec<_>>();

    let mut plt = Plot2D::new();
    plt.set_domain(x.clone());
    for grf in grf_scaled_vec.iter().take(samples) {
        plt.insert_image(grf.clone());
    }
    plt
        .set_path("grf_scaled.png")
        .set_xlabel(r"$x$")
        .set_ylabel(r"$y$")
        .set_style(PlotStyle::Nature)
        .set_line_style(line_style.clone())
        .set_color(color.clone())
        .set_legend(legends_str.clone())
        .tight_layout()
        .set_dpi(600)
        .savefig()
        .unwrap();

    let mut plt = Plot2D::new();
    plt.set_domain(y_range);
    for sol in grf_int_vec.iter().take(samples) {
        plt.insert_image(sol.clone());
    }
    plt
        .set_path("grf_integral.png")
        .set_xlabel(r"$x$")
        .set_ylabel(r"$y$")
        .set_style(PlotStyle::Nature)
        .set_line_style(line_style.clone())
        .set_color(color.clone())
        .set_legend(legends_str.clone())
        .tight_layout()
        .set_dpi(600)
        .savefig()
        .unwrap();
}
