"""Prospective Phase 9 arithmetic, selection and validation helpers."""
from fractions import Fraction


def pointwise_backbone(tf, backbone):
    """Build a VJP-only view that shares variables without mutating the source."""
    from .phase8_training import pointwise_matmul

    class PointwiseView(tf.keras.layers.Layer):
        def __init__(self, source):
            super().__init__(name=source.name, dtype=source.dtype_policy)
            self.source = source

        def call(self, inputs):
            result = pointwise_matmul(tf, inputs, self.source.kernel)
            if self.source.use_bias:
                result = tf.nn.bias_add(result, self.source.bias)
            return self.source.activation(result)

    def share(layer):
        if (isinstance(layer, tf.keras.layers.Conv2D) and layer.trainable
                and layer.kernel_size == (1, 1) and layer.strides == (1, 1)
                and layer.dilation_rate == (1, 1) and layer.groups == 1
                and layer.data_format == 'channels_last' and layer.padding in ('same', 'valid')):
            return PointwiseView(layer)
        return layer

    view = tf.keras.models.clone_model(backbone, clone_function=share)
    if {id(v) for v in view.trainable_weights} != {id(v) for v in backbone.trainable_weights}:
        raise ValueError('Gradient view must share exactly the trainable variables')
    return view


def guarded_selection(counts, reference, margins):
    """Screen matched count denominators and minimize balanced stress risk.

    All fractions use exact integer ratios. Margins are prospective policy
    strings, for example '0.02'. This is an empirical selection rule.
    """
    guards = ('A_clean_FA', 'A_clean_miss', 'B_clean_FA', 'A05_stress_miss', 'A06_stress_miss')
    risks = guards + ('A_stress_FA', 'A_stress_miss')
    if set(counts) != set(risks) or set(reference) != set(risks) or set(margins) != set(guards):
        raise ValueError('Missing or unexpected registered group')
    deltas = {}
    for key in risks:
        errors, denominator = counts[key]
        ref_errors, ref_denominator = reference[key]
        if (any(type(x) is not int for x in (errors, denominator, ref_errors, ref_denominator))
                or denominator != ref_denominator or denominator <= 0
                or not 0 <= errors <= denominator or not 0 <= ref_errors <= denominator):
            raise ValueError(f'Invalid matched denominator for {key}')
        deltas[key] = Fraction(errors - ref_errors, denominator)
    if any(not 0 <= Fraction(value) <= 1 for value in margins.values()):
        raise ValueError('Margins must be in [0, 1]')
    failed = [key for key in guards if deltas[key] > Fraction(margins[key])]
    if failed:
        return {'status': 'NO_FEASIBLE_CHECKPOINT', 'failed': failed, 'objective': None}
    stress = sum(Fraction(*counts[key]) for key in risks[-2:]) / 2
    return {'status': 'ELIGIBLE', 'failed': [], 'objective': float(stress)}
