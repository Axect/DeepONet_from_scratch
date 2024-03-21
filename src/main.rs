use rugfield::gen_grf;
use peroxide::fuga::*;

fn main() {
    let x_min = 0f64;
    let x_max = 1f64;

    let grf_vec = (0 .. 1000)
        .map(|_| gen_grf(x_min, x_max, 0.1, 1000))
        .collect::<Vec<_>>();

    let grf_max_vec = grf_vec
        .iter()
        .map(|grf| grf.max())
        .collect::<Vec<_>>();

    let grf_min_vec = grf_vec
        .iter()
        .map(|grf| grf.min())
        .collect::<Vec<_>>();

    let grf_scaled = grf_vec.iter().zip(grf_max_vec.iter()).zip(grf_min_vec.iter())
        .map(|((grf, max), min)| {
            grf.fmap(|x| (x - min) / (max - min))
        }).collect::<Vec<_>>();

    let x = linspace_with_precision(x_min, x_max, 1000, 3);

    let samples = 10;
    let line_style = [LineStyle::Solid, LineStyle::Dotted, LineStyle::Dashed, LineStyle::DashDot];
    let line_style = line_style.iter().cycle().take(samples).cloned().collect::<Vec<_>>();
    let color = ["darkblue", "red", "darkgreen", "darkorange", "purple"];
    let color = color.iter().cycle().take(samples).cloned().collect::<Vec<_>>();

    let mut plt = Plot2D::new();
    plt.set_domain(x);
    for grf in grf_scaled.iter().take(samples) {
        plt.insert_image(grf.clone());
    }
    plt
        .set_path("grf_scaled.png")
        .set_xlabel(r"$x$")
        .set_ylabel(r"$y$")
        .set_style(PlotStyle::Nature)
        .set_line_style(line_style)
        .set_color(color)
        .tight_layout()
        .set_dpi(600)
        .savefig()
        .unwrap();
}
