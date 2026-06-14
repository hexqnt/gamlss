use thiserror::Error;

/// Ошибки построения и проверки GAMLSS-моделей.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModelError {
    /// Response vector пуст.
    #[error("response vector must contain at least one observation")]
    EmptyResponse,

    /// Скалярный параметр модели имеет недопустимое значение.
    #[error("{parameter} must be {expected}")]
    InvalidParameter {
        /// Имя параметра.
        parameter: &'static str,
        /// Ожидаемый инвариант.
        expected: &'static str,
    },

    /// Dense matrix получила неверное число row-major значений.
    ///
    /// Число переданных значений `actual_values` не совпадает с `nrows * ncols`.
    #[error("design matrix has {actual_values} values, expected {expected_values}")]
    DesignSize {
        /// Ожидаемое число значений.
        expected_values: usize,
        /// Фактическое число значений.
        actual_values: usize,
    },

    /// Размеры design matrix не помещаются в `usize`.
    #[error("arithmetic overflow while computing {context}")]
    ArithmeticOverflow {
        /// Описание вычисляемого размера.
        context: &'static str,
    },

    /// Число строк design matrix не совпадает с длиной response.
    #[error(
        "{parameter} design has {actual_rows} rows, expected {expected_rows} rows from response"
    )]
    DesignRowMismatch {
        /// Имя или роль проверяемого параметра.
        parameter: &'static str,
        /// Ожидаемое число строк.
        expected_rows: usize,
        /// Фактическое число строк.
        actual_rows: usize,
    },

    /// Длина response не совпадает с ожидаемой.
    #[error("response length is {actual}, expected {expected}")]
    ResponseLength {
        /// Ожидаемая длина.
        expected: usize,
        /// Фактическая длина.
        actual: usize,
    },

    /// Длина beta-вектора не совпадает с числом коэффициентов модели.
    #[error("beta length is {actual}, expected {expected}")]
    BetaLength {
        /// Ожидаемая длина.
        expected: usize,
        /// Фактическая длина.
        actual: usize,
    },

    /// Длина gradient-вектора не совпадает с числом коэффициентов модели.
    #[error("gradient length is {actual}, expected {expected}")]
    GradientLength {
        /// Ожидаемая длина.
        expected: usize,
        /// Фактическая длина.
        actual: usize,
    },

    /// Индекс строки predictor-а вне диапазона наблюдений модели.
    #[error("row index {row} is out of bounds for {nrows} rows")]
    RowOutOfBounds {
        /// Запрошенный индекс строки.
        row: usize,
        /// Число строк в модели.
        nrows: usize,
    },

    /// Два parameter block используют пересекающиеся диапазоны beta.
    #[error("{first} parameter block overlaps with {second} parameter block")]
    BlockOverlap {
        /// Первый пересекающийся блок.
        first: &'static str,
        /// Второй пересекающийся блок.
        second: &'static str,
    },

    /// Диапазон коэффициентов parameter block не помещается в `usize`.
    #[error("{parameter} parameter block range overflows: offset {offset}, len {len}")]
    BlockRangeOverflow {
        /// Имя параметра.
        parameter: &'static str,
        /// Начальная позиция блока.
        offset: usize,
        /// Длина блока.
        len: usize,
    },
}
