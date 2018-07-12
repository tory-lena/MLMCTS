importall MCTS
import MCTS.simulate, MCTS.insert_action_node!, MCTS.insert_state_node!, MCTS.init_N, MCTS.init_Q, MCTS.next_action

function MCTS.simulate(dpw::DPWPlanner, snode::Int, d::Int)
    if d == 0 || isterminal(dpw.mdp, s)
        return 0.0
    end
    l=6; i=1
    Q=0.
    while l>2 && i<=5
        i+=1
        q, l = simulation_loop(dpw, snode, d)
        Q+=q
    end
    return Q/i
end
    
function simulation_loop(dpw::DPWPlanner, snode::Int, d::Int)
    
    #print(fieldnames(dpw.tree))
    S = state_type(dpw.mdp)
    A = action_type(dpw.mdp)
    sol = dpw.solver
    tree = get(dpw.tree)
    s = tree.s_labels[snode]
    if d == 0 || isterminal(dpw.mdp, s)
        return 0.0
    end

    # action progressive widening
    if dpw.solver.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(dpw.next_action, dpw.mdp, s, DPWStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, dpw.mdp, s, a)
                insert_action_node!(tree, snode, a, n0,
                                    init_Q(sol.init_Q, dpw.mdp, s, a),
                                    sol.check_repeat_action
                                   )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        for a in iterator(actions(dpw.mdp, s))
            n0 = init_N(sol.init_N, dpw.mdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, dpw.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end

    best_UCB = -Inf
    sanode = 0
    ltn = log(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        if ltn <= 0 && n == 0
            UCB = q
        else
            c = sol.exploration_constant # for clarity
            UCB = q + c*sqrt(ltn/n)
        end
        @assert !isnan(UCB)
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            sanode = child
        end
    end

    a = tree.a_labels[sanode]

    # state progressive widening
    new_node = false
    if true #tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state
        sp, r , l= generate_sr(dpw.mdp, s, a, dpw.rng)

        spnode = sol.check_repeat_state ? get(tree.s_lookup, sp, 0) : 0

        if spnode == 0 # there was not a state node for sp already in the tree
            spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end
        push!(tree.transitions[sanode], (spnode, r))

        if !sol.check_repeat_state 
            tree.n_a_children[sanode] += 1
        #elseif !((sanode,spnode) in tree.unique_transitions)
        #    push!(tree.unique_transitions, (sanode,spnode))
        #    tree.n_a_children[sanode] += 1
        end
    else
        spnode, r = rand(dpw.rng, tree.transitions[sanode])
    end

    if !new_node
        q = r + discount(dpw.mdp)*estimate_value(dpw.solved_estimate, dpw.mdp, sp, d-1)
    else
        q = r + discount(dpw.mdp)*MCTS.simulate(dpw, spnode, d-1)
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    return q, l
end