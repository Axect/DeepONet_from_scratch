use rugfield::gen_grf;
use peroxide::fuga::*;
use rayon::prelude::*;
use indicatif::{ProgressBar, ParallelProgressIterator};

fn main() {
    // Gaussian Random Field generation
    let x_min = 0f64;
    let x_max = 1f64;
    let n = 10000;
    let x_len = 1000;
    //let l_uniform = WeightedUniform::new(
    //    vec![3f64, 3f64, 2f64, 2f64],
    //    vec![0.1, 0.2, 0.3, 0.4, 0.5],
    //);
    let l_uniform = Uniform(0.1, 0.4);
    let l_samples = l_uniform.sample(n);

    let grf_vec = l_samples
        .iter()
        .map(|&l| gen_grf(x_min, x_max, l, x_len))
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

    let x = linspace_with_precision(x_min, x_max, x_len, 3);
    let u_y = Uniform(x_min, x_max);
    let y_range = u_y.sample(100);
    
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

    // Save parquet
    let mut df = DataFrame::new(vec![]);
    let x_cycle = x.iter().cycle().take(n * x.len()).cloned().collect::<Vec<_>>();
    let grf_flatten = grf_scaled_vec.iter().flatten().cloned().collect::<Vec<_>>();
    let group = (0u64 .. n as u64).flat_map(|i| std::iter::repeat(i).take(x_len)).collect::<Vec<_>>();
    df.push("x", Series::new(x_cycle));
    df.push("grf", Series::new(grf_flatten));
    df.push("group", Series::new(group));
    df.write_parquet("data/grf_random_l.parquet", CompressionOptions::Uncompressed).unwrap();
    df.print();

    let mut dg = DataFrame::new(vec![]);
    let y_cycle = y_range.iter().cycle().take(n * y_range.len()).cloned().collect::<Vec<_>>();
    let grf_int_flatten = grf_int_vec.iter().flatten().cloned().collect::<Vec<_>>();
    let group = (0u64 .. n as u64).flat_map(|i| std::iter::repeat(i).take(y_range.len())).collect::<Vec<_>>();
    dg.push("y", Series::new(y_cycle));
    dg.push("grf_int", Series::new(grf_int_flatten));
    dg.push("group", Series::new(group));
    dg.write_parquet("data/grf_random_l_int.parquet", CompressionOptions::Uncompressed).unwrap();
    dg.print();

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
        .set_path("figs/grf_random_l_scaled.png")
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

    let mut y_range_enum = y_range.into_iter().enumerate().collect::<Vec<_>>();
    y_range_enum.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let (y_idx, y_range): (Vec<usize>, Vec<f64>) = y_range_enum.into_iter().unzip();

    let mut plt = Plot2D::new();
    plt.set_domain(y_range);
    for sol in grf_int_vec.iter().take(samples) {
        let sol = y_idx.iter().map(|&i| sol[i]).collect::<Vec<_>>();
        plt.insert_image(sol.clone());
    }
    plt
        .set_path("figs/grf_random_l_integral.png")
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
