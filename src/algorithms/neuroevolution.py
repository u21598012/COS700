
from src.algorithms.base_algorithm import BaseAlgorithm, SearchOutcome
import copy
import random
import time

from src.evaluation.evaluator import evaluate_architecture
from src.models.architecture import ArchitectureConfig

class Neuroevolution(BaseAlgorithm):
    name = "neuroevolution"

    def __init__(self, search_space, budget, selection_metric="macro_f1"):
        super().__init__(search_space, budget, selection_metric)
       
        self.population = []
        self.population_metadata = {}
        self.best_architecture = None
        
        self.population_size = budget
       
        self.verbose = False
        self._reset_tracking()

    def _reset_tracking(self):
        self._outcome = SearchOutcome()
        self._eval_count = 0
        self._best_score = float("-inf")
        self._start = time.perf_counter()

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def generate_population(self, population_size):
       
        self.population_size = population_size
        self.population = [self.search_space.sample_architecture() for _ in range(population_size)]

    def evaluate_architecture(self, architecture, evaluate_fn, on_evaluation):
        result = evaluate_fn(architecture)

      
        self._eval_count += 1
        eval_number = self._eval_count
        elapsed = time.perf_counter() - self._start
        score = self._score(result)
        if score > self._best_score:
            self._best_score = score
            self._outcome.best_architecture = architecture
            self._outcome.best_result = result
            self._outcome.best_evaluation_number = eval_number
            self._outcome.time_to_best = elapsed
        self._outcome.history.append({
            "evaluation_number": eval_number,
            "score": score,
            "elapsed": elapsed,
        })

        if on_evaluation:
            on_evaluation(eval_number, architecture, result)
        return result
    
    def evaluate_population(self, evaluate_fn, on_evaluation):
        for architecture in self.population:
            fitness =self.evaluate_architecture(architecture, evaluate_fn, on_evaluation)
            architecture_metadata = {
                "id": id(architecture),
                "architecture": architecture,
                "fitness": fitness,
            }
            self.population_metadata[id(architecture)] = architecture_metadata

    def evaluate_and_register(self, architecture, evaluate_fn, on_evaluation=None):
      
        existing = self.population_metadata.get(id(architecture))
        if existing is not None and existing["fitness"] is not None:
            return existing["fitness"]
        fitness = self.evaluate_architecture(architecture, evaluate_fn, on_evaluation)
        self.population_metadata[id(architecture)] = {
            "id": id(architecture),
            "architecture": architecture,
            "fitness": fitness,
        }
        return fitness

    def select_best_architecture(self):
        best_fitness = float("-inf")
        best_architecture = None

        if self.best_architecture is not None and id(self.best_architecture) in self.population_metadata:
            best_fitness = self.population_metadata[id(self.best_architecture)]["fitness"].macro_f1
            best_architecture = self.best_architecture

        for architecture in self.population:
            fitness = self.population_metadata[id(architecture)]["fitness"].macro_f1  # selection metric
            if fitness > best_fitness:
                best_fitness = fitness
                best_architecture = architecture
        self.best_architecture = best_architecture

    def roulette_wheel_selection(self, num_selections):
        candidates = [self.population_metadata[id(a)] for a in self.population]
        selected = []
        for _ in range(num_selections):
            total_fitness = sum(metadata["fitness"].macro_f1 for metadata in candidates)
            p = random.uniform(0, total_fitness)
            current = 0
            for metadata in candidates:
                current += metadata["fitness"].macro_f1
                if current >= p:
                    selected.append(metadata["architecture"])
                    candidates.remove(metadata)
                    break

        return selected
    
    def tournament_selection(self, num_selections):
        candidates = [self.population_metadata[id(a)] for a in self.population]
        selected = []
        for _ in range(num_selections):
            tournament_size = min(4, len(candidates))  
            tournament_candidates = random.sample(candidates, tournament_size)
            best_candidate = max(tournament_candidates, key=lambda metadata: metadata["fitness"].macro_f1)
            selected.append(best_candidate["architecture"])
            candidates.remove(best_candidate)

        return selected

    def mutate_architecture(self, architecture):
        mutation_type = random.choice(["global", "layer"])
        if mutation_type == "global":
            architecture = self.global_mutation(architecture)
        elif mutation_type == "layer":
            architecture = self.inner_layer_mutation(architecture)
        return architecture

    def inner_layer_mutation(self, architecture):
        arch_ = copy.deepcopy(architecture)
        layer_index = random.randint(0, architecture.num_layers - 1)
        layer = arch_.layers[layer_index]
        ss = self.search_space

        while layer == architecture.layers[layer_index]:
            if layer.layer_type == "CNN":
                feature = random.choice(["hidden_size", "dropout", "activation", "kernel_size", "pooling"])
            else:  # LSTM / GRU
                feature = random.choice(["hidden_size", "dropout", "activation", "bidirectional", "pooling"])

            if feature == "hidden_size":
                layer.hidden_size = ss.sample_hidden_size()
            elif feature == "dropout":
                layer.dropout = ss.sample_dropout()
            elif feature == "activation":
                layer.activation = ss.sample_activation()
            elif feature == "kernel_size":
                layer.kernel_size = ss.sample_kernel_size()
            elif feature == "bidirectional":
                layer.bidirectional = ss.sample_bidirectional()
            elif feature == "pooling":
                layer.pooling = ss.sample_pooling()

        return arch_

    def create_layer(self, exclude_type=None):
      
        layer = self.search_space.sample_layer()
        choices = self.search_space.space["layer"]["type"]["choices"]
        if exclude_type is not None and len(choices) > 1:
            while layer.layer_type == exclude_type:
                layer = self.search_space.sample_layer()
        return layer

    def global_mutation(self, architecture):
        arch_ = copy.deepcopy(architecture)
        bounds = self.search_space.space["global"]["num_layers"]

        if random.random() < 0.5:
            arch_.learning_rate = self.search_space.sample_learning_rate()
        else:
          
            updates = []
            if arch_.num_layers > 0:
                updates.append("replace")
            if arch_.num_layers < bounds["max"]:
                updates.append("create")
            if arch_.num_layers > bounds["min"]:
                updates.append("delete")
            update = random.choice(updates)

            if update == "create":
                arch_.layers.append(self.create_layer())
                arch_.num_layers += 1
            elif update == "replace":
                layer_index = random.randint(0, arch_.num_layers - 1)
                current_type = arch_.layers[layer_index].layer_type
                arch_.layers[layer_index] = self.create_layer(exclude_type=current_type)
            elif update == "delete":
                layer_index = random.randint(0, arch_.num_layers - 1)
                del arch_.layers[layer_index]
                arch_.num_layers -= 1

        return arch_
    
    def single_point_crossover(self, parent1, parent2):
        min_layers = min(parent1.num_layers, parent2.num_layers)
        point = random.randint(1, min_layers - 1) if min_layers > 1 else 1

        parent1_subset_1 = parent1.layers[:point]
        parent1_subset_2 = parent1.layers[point:]
        parent2_subset_1 = parent2.layers[:point]
        parent2_subset_2 = parent2.layers[point:]

       
        child1_layers = copy.deepcopy(parent1_subset_1 + parent2_subset_2)
        child2_layers = copy.deepcopy(parent2_subset_1 + parent1_subset_2)

        child_1 = ArchitectureConfig(
            learning_rate=parent1.learning_rate,
            num_layers=len(child1_layers),
            layers=child1_layers,
        )

        child_2 = ArchitectureConfig(
            learning_rate=parent2.learning_rate,
            num_layers=len(child2_layers),
            layers=child2_layers,
        )

        return child_1, child_2

    def uniform_crossover(self, parent1, parent2):
        min_layers = min(parent1.num_layers, parent2.num_layers)

        child1_layers = []
        child2_layers = []

        for l in range(min_layers):
            if random.random() < 0.5:
                child1_layers.append(parent1.layers[l])
                child2_layers.append(parent2.layers[l])
            else:
                child1_layers.append(parent2.layers[l])
                child2_layers.append(parent1.layers[l])

        child1_layers = copy.deepcopy(child1_layers)
        child2_layers = copy.deepcopy(child2_layers)

        child_1 = ArchitectureConfig(
            learning_rate=parent1.learning_rate,
            num_layers=len(child1_layers),
            layers=child1_layers,
        )

        child_2 = ArchitectureConfig(
            learning_rate=parent2.learning_rate,
            num_layers=len(child2_layers),
            layers=child2_layers,
        )

        return child_1, child_2
    
    def crossover(self, parent1, parent2):
        parent1_ = copy.deepcopy(parent1)
        parent2_ = copy.deepcopy(parent2)
        offspring = []

        crossover_type = random.choice(["single_point", "uniform"])

        if crossover_type == "single_point":
            self._vprint("Single-point crossover selected.",'\n')
            child1, child2 = self.single_point_crossover(parent1_, parent2_)
            offspring.extend([child1, child2])
        elif crossover_type == "uniform":
            self._vprint("Uniform crossover selected.",'\n')
            child1, child2 = self.uniform_crossover(parent1_, parent2_)
            offspring.extend([child1, child2])

        return offspring
    
    def final_selection(self, candidates, num_selections, evaluate_fn, on_evaluation=None):
        for architecture in candidates:
            self.evaluate_and_register(architecture, evaluate_fn, on_evaluation)
        sorted_candidates = sorted(
            candidates,
            key=lambda arch: self.population_metadata[id(arch)]["fitness"].macro_f1,
            reverse=True,
        )
        
        best_candidates = sorted_candidates[:num_selections // 2]
        if best_candidates[0] == best_candidates[1]:
          
            best_candidates[1] = sorted_candidates[num_selections // 2]

        worst_candidates = sorted_candidates[-(num_selections // 2):]
        if worst_candidates[0] == worst_candidates[1]:
           
            worst_candidates[1] = sorted_candidates[-(num_selections // 2 + 1)]

        return best_candidates + worst_candidates
    
    def steady_state_replacement(self, num_selections, candidates):
        for architecture in candidates:
            if architecture not in self.population:
                metadata = self.population_metadata[id(architecture)]
                self._vprint(f"Adding architecture ID: {id(architecture)} to population, Architecture: {architecture}, Fitness: {metadata['fitness']}")
                self.population.append(architecture)

      
        while len(self.population) > self.population_size:
            weakest = min(
                self.population,
                key=lambda a: self.population_metadata[id(a)]["fitness"].macro_f1,
            )
            metadata = self.population_metadata[id(weakest)]
            self._vprint(f"Removing architecture ID: {id(weakest)} from population, Architecture: {weakest}, Fitness: {metadata['fitness']}")
            self.population.remove(weakest)
            self.population_metadata.pop(id(weakest), None)

       
        alive = {id(a) for a in self.population}
        for key in [k for k in self.population_metadata if k not in alive]:
            self.population_metadata.pop(key)

    def execute(self, number_of_generations, num_selections, mutation_rate, crossover_rate, population_size, evaluate_fn, on_evaluation=None):
        self._reset_tracking()
        self.generate_population(population_size)
        self.evaluate_population(evaluate_fn, on_evaluation)
        self.select_best_architecture()

        for generation in range(number_of_generations):
            if self._eval_count >= self.budget:
                break

            self._vprint(f"Generation {generation + 1}/{number_of_generations}")
            self._vprint('-'*80)
            self._vprint('\n')

            for a in self.population:
                self._vprint(f"Population Architecture ID: {id(a)}, Architecture: {a}, Fitness: {self.population_metadata[id(a)]['fitness']}")
                self._vprint('\n')
                self._vprint('-'*80)
            self._vprint('\n')
            subset = self.roulette_wheel_selection(num_selections)

            mutated_parents = []

            for parent in subset:
                self._vprint("Subset Selection:",'\n')
                self._vprint(f"Parent Architecture ID: {id(parent)}, Architecture: {parent}, Fitness: {self.population_metadata[id(parent)]['fitness']}")
                self._vprint('\n')
            self._vprint('\n')

            self._vprint("="*80,'Subset Selection Complete',"="*80)

            for parent in subset:
                self._vprint("Considering Parent for Mutation:",'\n')
                self._vprint(f"Parent Architecture ID: {id(parent)}, Architecture: {parent}, Fitness: {self.population_metadata[id(parent)]['fitness']}")
                self._vprint('\n')
                if random.random() < mutation_rate:
                    self._vprint("Mutation applied to parent:",'\n')
                    mutated_parent = self.mutate_architecture(parent)
                    self._vprint("mutation comparison:",'\n')
                    self._vprint(f"Parent Architecture ID: {id(parent)}, Architecture: {parent}")
                    self._vprint('-'*80)
                    self._vprint(f"Mutated Parent Architecture ID: {id(mutated_parent)}, Architecture: {mutated_parent}")
                    self._vprint('\n')
                    mutated_parents.append(mutated_parent)
                else:
                    self._vprint("Mutation skipped for parent:",'\n')
                    mutated_parents.append(parent)

            for im in mutated_parents:
                self._vprint(f"Mutated Architecture ID: {id(im)} added to mutated parents, Architecture: {im}")
                self._vprint('\n')

            self._vprint("="*80,'Mutation Complete',"="*80)

            self._vprint("Crossover Phase:",'\n')

            offspring = []
            crossover_pairs = [(mutated_parents[i], mutated_parents[i + 1]) for i in range(0, len(mutated_parents) - 1, 2)]
            for pair in crossover_pairs:
                if random.random() < crossover_rate:
                    self._vprint("Crossover Comparison:",'\n')
                    self._vprint(f"Parent 1 Architecture ID: {id(pair[0])} selected for crossover, Architecture: {pair[0]}")
                    self._vprint('-'*80)
                    self._vprint(f"Parent 2 Architecture ID: {id(pair[1])} selected for crossover, Architecture: {pair[1]}")
                    self._vprint('\n')
                    children = self.crossover(pair[0], pair[1])
                    self._vprint("Crossover Results:\n")
                    for child in children:
                        self._vprint(f"Child Architecture ID: {id(child)} created from crossover, Architecture: {child}")
                        self._vprint('\n')
                    offspring.extend(children)
                else:
                    offspring.extend(pair)

          
            if len(mutated_parents) % 2 == 1:
                offspring.append(mutated_parents[-1])

            self._vprint("="*80,'Crossover Complete',"="*80)

            evaluated = []
            for architecture in offspring:
                scored = self.population_metadata.get(id(architecture))
                needs_eval = scored is None or scored["fitness"] is None
                if needs_eval and self._eval_count >= self.budget:
                    continue
                self.evaluate_and_register(architecture, evaluate_fn, on_evaluation)
                evaluated.append(architecture)

            for t in evaluated:
                self._vprint(f"Population Update Architecture ID: {id(t)}")
                self._vprint('\n')

            self.steady_state_replacement(num_selections, evaluated)
            self.select_best_architecture()

            self._vprint("="*80)
            self._vprint(f"{time.perf_counter() - self._start} seconds elapsed")

        self._outcome.total_search_time = time.perf_counter() - self._start
        return self._outcome

    def search(self, evaluate_fn, on_evaluation=None, verbose=False):
        self.verbose = verbose
      
        population_size = min(self.budget, max(2, int(self.budget * 0.20)))
        generation_size = max(1, int((self.budget * 0.80) / 4))
        return self.execute(
            number_of_generations=generation_size,
            num_selections=4,
            mutation_rate=0.5,
            crossover_rate=0.5,
            population_size=population_size,
            evaluate_fn=evaluate_fn,
            on_evaluation=on_evaluation
        )


if __name__ == "__main__":
    print("main")
    # import sys

    # from src.models.search_space import SearchSpace
    # from src.experiment.experiment_manager import ExperimentManager

    # class _Tee:
    #     def __init__(self, *streams):
    #         self.streams = streams

    #     def write(self, data):
    #         for s in self.streams:
    #             s.write(data)

    #     def flush(self):
    #         for s in self.streams:
    #             s.flush()

    # output_path = "neuroevolution_output.txt"
    # log_file = open(output_path, "w", encoding="utf-8")
    # sys.stdout = _Tee(sys.__stdout__, log_file)

    # try:
    #     experiment_manager = ExperimentManager()

    #     loaders, meta = experiment_manager.load_data()

      

    #     counter = {"n": 0}

    #     def evaluate_fn(architecture):
    #         counter["n"] += 1
    #         print(f"[eval {counter['n']}] training {architecture.num_layers}-layer model...", flush=True)
    #         return evaluate_architecture(
    #             architecture, loaders["train"], loaders["val"],
    #             experiment_manager.experiment_cfg, meta["input_dim"],
    #             meta["num_classes"], device=experiment_manager.device,
    #         )

    #     def on_evaluation(eval_number, architecture, result):
    #         status = f"FAILED: {result.error}" if result.failed else f"macro_f1={result.macro_f1:.4f} acc={result.accuracy:.4f}"
    #         print(f"    -> {status}  ({result.training_time:.1f}s train)", flush=True)

    #     search_space = SearchSpace.from_yaml()
    #     neuroevolution = Neuroevolution(search_space, budget=50)
    #     outcome = neuroevolution.search(evaluate_fn, on_evaluation, verbose=True)
    #     print(f"Best architecture (eval #{outcome.best_evaluation_number}): {outcome.best_architecture}")
    #     print(f"Best fitness: {outcome.best_result}")
    #     print(f"Total evaluations: {len(outcome.history)}, total search time: {outcome.total_search_time:.1f}s")
    # finally:
    #     sys.stdout = sys.__stdout__
    #     log_file.close()
    #     print(f"Output written to {output_path}")

