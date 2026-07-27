datasets! {
    a1 {
        path: "data/a1.csv",
        x: f64,
        y: f64,
    }
    faithful {
        path: "data/faithful.csv",
        x: u16,
        y: [f64; 2],
    }
    cbr_inflation_and_interest_rate {
        path: "data/cbr_inflation_and_interest_rate.csv",
        x: Date,
        y: [f64; 2],
    }
}
