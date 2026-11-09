"""Single-mechanism ablations; full preserves the existing stable policy."""
from dataclasses import asdict, dataclass


VARIANTS = ('full', 'fixed_knobs', 'fixed_regions', 'uniform_budget', 'nei_only')


@dataclass(frozen=True)
class Ablation:
    name: str = 'full'

    def __post_init__(self):
        if self.name not in VARIANTS:
            raise ValueError('unknown ablation: '+self.name)

    @property
    def dynamic_knobs(self): return self.name != 'fixed_knobs'

    @property
    def dynamic_regions(self): return self.name != 'fixed_regions'

    @property
    def adaptive_allocation(self): return self.name != 'uniform_budget'

    @property
    def information_selection(self): return self.name != 'nei_only'

    def describe(self):
        return {**asdict(self), **{key:getattr(self,key) for key in
            ('dynamic_knobs','dynamic_regions','adaptive_allocation','information_selection')}}


def project_patch(patch, expert_regions, space, unit, value_at):
    """Project an explicit proposal into frozen recommended support.

    Parameter groups are retained; invalidated joint semantics must instead be
    rejected before owner agreement. Global reference/exploration is unchanged.
    """
    result = dict(patch)
    for name, value in patch.items():
        if name not in expert_regions:
            continue
        knob, region = space.by_name[name], expert_regions[name]
        if knob.kind in {'integer','float'}:
            lo,hi=max(0,region['center']-region['halfwidth']),min(1,region['center']+region['halfwidth'])
            coordinate=unit(knob,value)
            if coordinate < lo or coordinate > hi:
                result[name]=value_at(knob,min(hi,max(lo,coordinate)))
        else:
            choices=region['values']
            if value not in choices:
                result[name]=choices[0]
    return result
